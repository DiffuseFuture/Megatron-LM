# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT."""

import datetime
import os
import torch
import copy
from torch.utils.data import RandomSampler
from omegaconf import OmegaConf
from modelscope import AutoTokenizer
from einops import rearrange
import numpy as np
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
import torchvision.transforms.functional as TF
from PIL import Image
from megatron.core.packed_seq_params import PackedSeqParams




from functools import partial
from typing import List, Optional, Tuple, Union
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from dataset_image_video import ImageVideoDataset, ImageVideoSampler

from megatron.core.enums import ModelType
from megatron.core.transformer.module import Float16Module

from megatron.core.models.VideoX_Fun.transformer3d_layer_specs import (
    get_transformer3d_layer_local_spec,
    get_transformer3d_transformer_engine_block_spec,

)

from megatron.core.models.T5.t5_spec import (
    get_t5_encoder_with_transformer_engine_block_spec,
    get_t5_encoder_with_local_block_spec,
)

from megatron.core.models.VideoX_Fun import WanTransformer3DModel
from megatron.core.models.T5 import T5Model
from megatron.core.models.wan.wan_vae import WanVae, AutoencoderKLWan
from megatron.core.models.wan.wan_image_encoder import WanCLIP, get_wan_clip_spec, get_wan_clip_spec_te

from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer3d_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml

import megatron.legacy.model  # isort: skip
from utils import resize_mask, get_timesteps_and_sigmas, filter_kwargs

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


t5_model = None
dit_model = None
tokenizer = None
noise_scheduler = None
vae_model = None
clip_model = None





def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[WanTransformer3DModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    wan_config = OmegaConf.load("wan_civitai.yaml")

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    
    print_rank_0('building WanTransformer3D model ...')
    # Experimental loading arguments from yaml
    config = core_transformer3d_config_from_args(args)

    if parallel_state.is_pipeline_first_stage():

        global t5_model, tokenizer, noise_scheduler, vae_model, clip_model
        
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join("/root/add_dit/models/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP", 
            wan_config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )

        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(wan_config['scheduler_kwargs']))
        )
        
        
        t5_config = copy.deepcopy(config)
        encoder_config = copy.deepcopy(config)

        en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(
            config.num_layers
        )

        
        t5_model = T5Model(
            config=t5_config,
            encoder_config=encoder_config,
            transformer_encoder_layer_spec=en_block_spec,
            transformer_decoder_layer_spec=None,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            relative_attention_num_buckets=args.relative_attention_num_buckets,
            relative_attention_max_distance=args.relative_attention_max_distance,
            add_encoder=True,
            add_decoder=False,
        )
        t5_model.cuda(torch.cuda.current_device())
        t5_model = Float16Module(t5_config, t5_model)
        t5_model.requires_grad_(False)

        # init vae
        vae_config = copy.deepcopy(config)
        
        pretrained_model_path = "/root/add_dit/models/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth" 
        vae_model = WanVae(config, pretrained_model_path).to(torch.float16).cuda(torch.cuda.current_device())
        vae_model.requires_grad_(False)
        # init clip
        clip_config = copy.deepcopy(config)
        clip_config.hidden_size = 1280
        clip_config.num_attention_heads = 8
        if args.transformer_impl == "transformer_engine":
            clip_spec = get_wan_clip_spec_te()
        else:
            clip_spec = get_wan_clip_spec()
        clip_model = WanCLIP(clip_config, clip_spec).to(torch.float16).cuda(torch.cuda.current_device())
        clip_model.requires_grad_(False)
        

    
    transformer_layer_spec = get_transformer3d_transformer_engine_block_spec()

    global dit_model
    dit_model = WanTransformer3DModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=None,
        vp_stage=vp_stage,
    )

    # return [t5_model, dit_model]
    return dit_model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        return None, None, None, None, None

    batch = next(data_iterator)

    pixel_values = batch["pixel_values"]
    text = batch["text"]
    clip_pixel_values = batch["clip_pixel_values"]
    mask_pixel_values = batch["mask_pixel_values"]
    mask = batch["mask"]
    
    return pixel_values, text, clip_pixel_values, mask_pixel_values, mask

# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[WanTransformer3DModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    # loss_mask = loss_mask.view(-1).float()
    # loss = torch.sum(losses * loss_mask)
    loss = torch.sum(losses)

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


def forward_step(data_iterator, model: WanTransformer3DModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    print("start fwd")
    args = get_args()
    timers = get_timers()
    weight_dtype = torch.float16

    
    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        # tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
        pixel_values, text, clip_pixel_values, mask_pixel_values, mask = get_batch(data_iterator)
        if pixel_values is not None:
            pixel_values = pixel_values.to(weight_dtype).cuda()
            clip_pixel_values = clip_pixel_values.to(weight_dtype).cuda()
            mask_pixel_values = mask_pixel_values.to(weight_dtype).cuda()
            mask = mask.to(weight_dtype).cuda()
        
    timers('batch-generator').stop()

    with stimer:
        if parallel_state.is_pipeline_first_stage():
            device = pixel_values.device
            pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
            mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
            
            # latents = vae.encoder(pixel_values)
            print("before vae pixel_values shape", pixel_values.shape)
            latents = vae_model.encode(pixel_values)[0].sample()
            print("after vae latents shape", latents.shape)
    
            mask = rearrange(mask, "b f c h w -> b c f h w")
            mask = torch.concat(
                [
                    torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                    mask[:, :, 1:]
                ], dim=2
            )
            mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
            mask = mask.transpose(1, 2)
            mask = resize_mask(1 - mask, latents)
            mask_latents = vae_model.encode(mask_pixel_values)[0].sample()
    
            t2v_flag = [(_mask == 1).all() for _mask in mask]
            new_t2v_flag = []
            for _mask in t2v_flag:
                if _mask and np.random.rand() < 0.90:
                    new_t2v_flag.append(0)
                else:
                    new_t2v_flag.append(1)
            t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).cuda()
    
            
            inpaint_latents = torch.concat([mask, mask_latents], dim=1)
            inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents

            # clip_mask = torch.zeros(1, 1, 257,257).to(torch.bool).cuda()
            clip_mask = None
            print("clip_pixel_values", clip_pixel_values.shape)
            clip_context = []
            for clip_input in clip_pixel_values:
                clip_image = Image.fromarray(np.uint8(clip_input.float().cpu().numpy()))
                clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
                print("clip_image[:, None, :, :]", clip_image[:, None, :, :].size())
                clip_fea = clip_model([clip_image[:, None, :, :]], clip_mask)
                zero_init_clip_in = np.random.choice([True, False], p=[0.1, 0.9])
                clip_context.append(clip_fea if not zero_init_clip_in else torch.zeros_like(clip_fea))
                
            clip_context = torch.cat(clip_context)
            print("clip_context", clip_context.shape)
    
    
            # text
            args.tokenizer_max_length = 512
            prompt_ids = tokenizer(
                text, 
                padding="max_length", 
                max_length=args.tokenizer_max_length, 
                truncation=True, 
                add_special_tokens=True, 
                return_tensors="pt"
            )
            text_input_ids = prompt_ids.input_ids
            prompt_attention_mask = prompt_ids.attention_mask.cuda()


            seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
            print("seq_lens", seq_lens)
    
            b, s = prompt_attention_mask.size(0), prompt_attention_mask.size(1)
            # mask = torch.zeros((b, 1, s, s), device=latents.device, dtype = torch.bool)
            # print("prompt_attention_mask shape", prompt_attention_mask.shape)

            print("text_input_ids shape", text_input_ids.shape)
            context = t5_model(text_input_ids.cuda(), None, prompt_attention_mask, None, None)
            context = context.transpose(0, 1).contiguous()
            print("context shape", context.shape)
            context = [u[:v] for u, v in zip(context, seq_lens)]

            # noise
            noise = torch.randn(latents.size(), device=latents.device, generator=None, dtype=weight_dtype)
            timestep, sigmas = get_timesteps_and_sigmas(noise_scheduler, latents, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    
            target = noise - latents
            
            grid_sizes = []
            for noisy_latent in noisy_latents:
                grid_sizes.append([noisy_latent.size(1), noisy_latent.size(2) // 2, noisy_latent.size(3) // 2])

            hidden_state_seq_len = grid_sizes[0][0] * grid_sizes[0][1] * grid_sizes[0][2] 
            context_seqlen = seq_lens[0] + 257
            grid_sizes = torch.tensor(grid_sizes).cuda()

            q_mask = torch.zeros((1, hidden_state_seq_len), dtype=torch.bool).cuda()
            kv_mask = torch.zeros((1, 512), dtype=torch.bool).cuda()
            kv_mask_img = torch.zeros((1, 257), dtype=torch.bool).cuda()
            context_mask = (q_mask, kv_mask, kv_mask_img)
            # attn_mask = torch.zeros((hidden_state_seq_len, hidden_state_seq_len), dtype=torch.bool).cuda()
            attn_mask = None
            # attn_mask = torch.tensor([75600], dtype=torch.int32).cuda().reshape((-1, 1))
            # print()
            # cu_seqlens = torch.tensor([0, hidden_state_seq_len], dtype=torch.int32, device=device)

            # packed_seq_params = PackedSeqParams(
            #     qkv_format="thd",  # 如果你不清楚，先设为 None，后面可调整为 'thd' 或 'bshd'
            #     cu_seqlens_q=cu_seqlens,
            #     cu_seqlens_kv=cu_seqlens,
            #     cu_seqlens_q_padded=None,
            #     cu_seqlens_kv_padded=None,
            #     max_seqlen_q=torch.tensor(hidden_state_seq_len, dtype=torch.int32, device=device),
            #     max_seqlen_kv=torch.tensor(hidden_state_seq_len, dtype=torch.int32, device=device),
            # )
            # noisy_latents = noisy_latents.transpose(0, 1).contiguous()

            print("noisy_latents shape", noisy_latents.shape)
            print("timestep", timestep)
            for u in context:
                print("context shape", u.shape)
            print("noisy_latents shape", noisy_latents.shape)
            print("")
        else:
            noisy_latents = None
            context = None
            context_mask = None
            grid_sizes = None
            hidden_state_seq_len = None
            context_seqlen = None
            clip_fea = None
            timestep = torch.tensor([0]).cuda()
            inpaint_latents = None
            target = None
            attn_mask = None
            
        
        output_tensor = model(noisy_latents, timestep, context, context_mask, hidden_state_seq_len, 
                                context_seqlen, grid_sizes, attn_mask, clip_fea, inpaint_latents, target)
            
    

    
        # if(tokens is not None):
        #     device = tokens.device
        #     x = torch.randn(1, 16, 6, 90, 156, device=device, dtype=torch.float16)
        #     t = torch.tensor([0], device=device)  # 假设 t 是时间步长标记之类的
        #     context = [torch.randn(209, 512, device=device, dtype=torch.float16)]
        #     # context_mask =  torch.tril(torch.ones((1, 1, 21060, 209), device=context_ids.device, dtype=torch.bool))
        #     seqlen = 21060  # 这是个整数，不用放 device 上
        #     y = torch.randn(1, 20, 6, 90, 156, device=device, dtype=torch.float16)
        #     clip_fea = torch.randn(1, 257, 1280, device=device, dtype=torch.float16)
        # else:
        #     x = None
        #     t = torch.tensor([0]).cuda()  # 假设 t 是时间步长标记之类的
        #     context = None
        #     seqlen = None
        #     clip_fea = None
        #     y = None
        # if parallel_state.is_pipeline_first_stage():
        #     context_ids = torch.ones((1, 209), dtype=torch.int64).cuda()
        #     causal_mask = torch.tril(torch.ones((1, 209, 209), device=context_ids.device, dtype=torch.bool))
        #     t5_model.eval()
        #     context = t5_model(context_ids, None, causal_mask, None, None)
        #     context = context.transpose(0, 1).contiguous()
        #     context = [c for c in context]
        # q_mask = torch.zeros((1, 1, 1, 21060), dtype=torch.bool).cuda()
        # kv_mask = torch.zeros((1, 1, 1, 512), dtype=torch.bool).cuda()
        # kv_mask_img = torch.zeros((1, 1, 1, 257), dtype=torch.bool).cuda()
        # context_mask = (q_mask, kv_mask, kv_mask_img)
        # hidden_state_seq_len = 21060
        # context_seqlen = 769

        # grid_sizes = torch.tensor([[ 6, 45, 78]], dtype=torch.int32).cuda()
        # output_tensor = model(x, t, context, context_mask, hidden_state_seq_len, context_seqlen, grid_sizes, None, clip_fea, y)

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    # loss_mask = None
    loss_mask = torch.ones((1, 100000)).cuda()
    return output_tensor, partial(loss_func, loss_mask, model=model)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """


    if parallel_state.is_pipeline_first_stage():

        def worker_init_fn(_seed):
            _seed = _seed * 256
            def _worker_init_fn(worker_id):
                print(f"worker_init_fn with {_seed + worker_id}")
                np.random.seed(_seed + worker_id)
                random.seed(_seed + worker_id)
            return _worker_init_fn
    
        
        args = get_args()
    
        args.train_data_meta="/root/add_dit/test_data/test.json"
        args.train_data_dir="/root/add_dit/test_data"
        args.video_sample_size=960
        args.video_sample_stride=2
        args.video_sample_n_frames=81
        args.video_repeat=1
        args.image_sample_size=1024
        args.enable_bucket=False
        args.seed = 22
        
        train_dataset = ImageVideoDataset(
            args.train_data_meta, args.train_data_dir,
            video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
            video_repeat=args.video_repeat, 
            image_sample_size=args.image_sample_size,
            enable_bucket=args.enable_bucket, enable_inpaint=True,
        )
        
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        dp_size = parallel_state.get_data_parallel_world_size()
        mini_batch_size = args.micro_batch_size * dp_size
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, mini_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler, 
            persistent_workers=False,
            num_workers=0,
            # persistent_workers=True if args.dataloader_num_workers != 0 else False,
            # num_workers=args.dataloader_num_workers,
            worker_init_fn=worker_init_fn(args.seed)
        )

        return train_dataloader, train_dataloader, train_dataloader
    else:
        return None, None, None

if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
