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
import torch.nn.functional as F
import json

from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)

import torchvision.transforms.functional as TF
from PIL import Image
from megatron.core.packed_seq_params import PackedSeqParams
from bucket_sampler import (ASPECT_RATIO_512,
                            ASPECT_RATIO_RANDOM_CROP_512,
                            ASPECT_RATIO_RANDOM_CROP_PROB,
                            AspectRatioBatchImageVideoSampler,
                            RandomSampler, get_closest_ratio)

from dataset_image_video import (ImageVideoDataset,
                                                 ImageVideoSampler,
                                                 get_random_mask)

from torchvision import transforms
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
sigmas = None




def get_gpu_memory_usage(device=None):
    """
    返回当前设备显存的使用情况，包含已分配显存、保留显存和显存摘要。

    参数:
    - device: 要监控的 GPU 设备，默认为 None，表示使用当前设备。

    返回:
    - 返回显存使用情况的字符串。
    """
    if torch.cuda.is_available():
        # 获取当前设备，如果未指定设备，则使用当前设备
        device = device or torch.cuda.current_device()

        # 获取当前 GPU 的显存使用情况
        allocated_memory = torch.cuda.memory_allocated(device)  # 已分配显存
        reserved_memory = torch.cuda.memory_reserved(device)  # 保留显存

        # 转换为 MB
        allocated_memory_mb = allocated_memory / (1024 ** 2)
        reserved_memory_mb = reserved_memory / (1024 ** 2)

        # 格式化显存使用情况
        memory_info = (
            f"Device {device} - Memory Allocated (bytes): {allocated_memory}\n"
            f"Device {device} - Memory Reserved (bytes): {reserved_memory}\n"
            f"Device {device} - Memory Allocated (MB): {allocated_memory_mb:.2f} MB\n"
            f"Device {device} - Memory Reserved (MB): {reserved_memory_mb:.2f} MB\n"
        )

        # 获取显存的摘要
        memory_summary = torch.cuda.memory_summary(device, abbreviated=True)
        memory_info += f"Device {device} - Memory Summary: \n{memory_summary}"

        return memory_info
    else:
        return "CUDA is not available. No GPU found."

def modify_t5_config(t5_config, wan_config):

    t5_config.num_layers = wan_config['text_encoder_kwargs'].get('num_layers')
    t5_config.hidden_size = wan_config['text_encoder_kwargs'].get('dim')
    t5_config.num_attention_heads = wan_config['text_encoder_kwargs'].get('num_heads')
    t5_config.num_query_groups = t5_config.num_attention_heads
    t5_config.ffn_hidden_size = wan_config['text_encoder_kwargs'].get('dim_ffn')
    t5_config.hidden_dropout = wan_config['text_encoder_kwargs'].get('dropout')
    t5_config.attention_dropout = wan_config['text_encoder_kwargs'].get('dropout')
    t5_config.kv_channels =  wan_config['text_encoder_kwargs'].get('dim_attn') // wan_config['text_encoder_kwargs'].get('num_heads')
    

    return t5_config


def modify_transformer3d_config(transformer3d_config):
    args = get_args()
    with open(args.transformer3d_config_path, 'r') as f:
        config_json = json.load(f)

    transformer3d_config.hidden_size = config_json.get('dim')
    transformer3d_config.ffn_hidden_size = config_json.get('ffn_dim')
    transformer3d_config.freq_dim = config_json.get('freq_dim')
    transformer3d_config.in_dim = config_json.get('in_dim')
    transformer3d_config.num_attention_heads = config_json.get('num_heads')
    transformer3d_config.num_layers = config_json.get('num_layers')
    transformer3d_config.out_dim = config_json.get('out_dim')
    transformer3d_config.text_dim = config_json.get('text_dim')     
    transformer3d_config.text_len = config_json.get('text_len')     

    return transformer3d_config



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
    wan_config = OmegaConf.load(args.wan_civitai_path)

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
    print("args.num_query_groups", args.num_query_groups)
    config = core_transformer3d_config_from_args(args)
    print("config.num_query_groups", config.num_query_groups)
    config = modify_transformer3d_config(config)
    print("change config.num_query_groups", config.num_query_groups)

    if parallel_state.is_pipeline_first_stage():

        global t5_model, tokenizer, vae_model, clip_model
        
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(args.pretrained_model_path, 
            wan_config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )
        
        
        t5_config = copy.deepcopy(config)
        t5_config = modify_t5_config(t5_config, wan_config)
        t5_config.context_parallel_size = 1

        en_block_spec = get_t5_encoder_with_transformer_engine_block_spec(
            t5_config.num_layers
        )
        t5_model = T5Model(
            config=t5_config,
            encoder_config=t5_config,
            transformer_encoder_layer_spec=en_block_spec,
            transformer_decoder_layer_spec=None,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=config.text_len,
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
        vae_config.context_parallel_size = 1
        
        vae_model_path = args.pretrained_model_path + "/Wan2.1_VAE.pth"
        vae_model = WanVae(config, vae_model_path).to(torch.float16).cuda(torch.cuda.current_device())
        vae_model.requires_grad_(False)
        # init clip
        clip_config = copy.deepcopy(config)
        clip_config.hidden_size = 1280
        clip_config.num_attention_heads = 8
        clip_config.num_query_groups= 8
        clip_config.context_parallel_size = 1
        if args.transformer_impl == "transformer_engine":
            clip_spec = get_wan_clip_spec_te()
        else:
            clip_spec = get_wan_clip_spec()
        clip_model = WanCLIP(clip_config, clip_spec).to(torch.float16).cuda(torch.cuda.current_device())
        clip_model.requires_grad_(False)
        

    global noise_scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(wan_config['scheduler_kwargs']))
    )
    
    transformer_layer_spec = get_transformer3d_transformer_engine_block_spec()

    global dit_model
    config.text_dim = wan_config['text_encoder_kwargs'].get('dim')
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



    def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
        diff = noise_pred - target
        mse_loss = F.mse_loss(noise_pred, target, reduction='none')
        mask = (diff.abs() <= threshold).float()
        masked_loss = mse_loss * mask
        if weighting is not None:
            masked_loss = masked_loss * weighting
        final_loss = masked_loss.mean()
        return final_loss

    output = output_tensor[0]
    target = output_tensor[1]

    weighting = compute_loss_weighting_for_sd3(weighting_scheme=None, sigmas=sigmas)
    loss = custom_mse_loss(output.to(torch.float32), target.to(torch.float32), weighting.to(torch.float32))

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
    args = get_args()
    timers = get_timers()
    weight_dtype = torch.float16
    global sigmas

    
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
            latents = vae_model.encode(pixel_values)[0].sample()
    
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

            clip_mask = None
            clip_context = []
            for clip_input in clip_pixel_values:
                clip_image = Image.fromarray(np.uint8(clip_input.float().cpu().numpy()))
                clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device, weight_dtype)
                clip_fea = clip_model([clip_image[:, None, :, :]], clip_mask)
                zero_init_clip_in = np.random.choice([True, False], p=[0.1, 0.9])
                clip_context.append(clip_fea if not zero_init_clip_in else torch.zeros_like(clip_fea))
                
            clip_context = torch.cat(clip_context)
    
            # text
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
    
            b, s = prompt_attention_mask.size(0), prompt_attention_mask.size(1)

            context = t5_model(text_input_ids.cuda(), None, prompt_attention_mask, None, None)
            context = context.transpose(0, 1).contiguous()
            context = [u[:v] for u, v in zip(context, seq_lens)]

            # noise
            noise = torch.randn(latents.size(), device=latents.device, generator=None, dtype=weight_dtype)
            timestep, sigmas = get_timesteps_and_sigmas(noise_scheduler, args.micro_batch_size, n_dim=5, dtype=weight_dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
    
            target = noise - latents
            target.requires_grad_(False)
            
            grid_sizes = []
            for noisy_latent in noisy_latents:
                grid_sizes.append([noisy_latent.size(1), noisy_latent.size(2) // 2, noisy_latent.size(3) // 2])

            hidden_state_seq_len = grid_sizes[0][0] * grid_sizes[0][1] * grid_sizes[0][2] 
            context_seqlen = seq_lens[0] + 257
            grid_sizes = torch.tensor(grid_sizes, dtype=torch.float16, requires_grad=False).cuda()

            q_mask = torch.zeros((1, hidden_state_seq_len), dtype=torch.bool).cuda()
            kv_mask = torch.zeros((1, 512), dtype=torch.bool).cuda()
            kv_mask_img = torch.zeros((1, 257), dtype=torch.bool).cuda()
            context_mask = (q_mask, kv_mask, kv_mask_img)
            attn_mask = None


        else:
            noisy_latents = None
            context = None
            context_mask = None
            grid_sizes = None
            hidden_state_seq_len = None
            context_seqlen = None
            clip_fea = None
            timestep, sigmas = get_timesteps_and_sigmas(noise_scheduler, args.micro_batch_size, n_dim=5, dtype=weight_dtype)
            inpaint_latents = None
            target = None
            attn_mask = None
    
        # # 检查并输出每个输入的形状，如果不是 None
        # if noisy_latents is not None:
        #     print(f"noisy_latents shape: {noisy_latents.shape}")
        
        # if timestep is not None:
        #     print(f"timestep shape: {timestep.shape}")
        
        # if context is not None:
        #     for c in context:
        #         print(f"context shape: {c.shape}")
        
        # if context_mask is not None:
        #     print(f"context_mask shape: {context_mask[0].shape, context_mask[1].shape, context_mask[2].shape}")
        
        # if grid_sizes is not None:
        #     print(f"grid_sizes shape: {grid_sizes.shape}")
        
        # if attn_mask is not None:
        #     print(f"attn_mask shape: {attn_mask.shape}")
        
        # if clip_fea is not None:
        #     print(f"clip_fea shape: {clip_fea.shape}")
        
        # if inpaint_latents is not None:
        #     print(f"inpaint_latents shape: {inpaint_latents.shape}")
        
        # if target is not None:
        #     print(f"target shape: {target.shape}")

        
        output_tensor = model(noisy_latents, timestep, context, context_mask, hidden_state_seq_len, 
                                context_seqlen, grid_sizes, attn_mask, clip_fea, inpaint_latents, target)

    loss_mask = torch.ones((1, 1)).cuda()
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
                np.random.seed(_seed + worker_id)
                random.seed(_seed + worker_id)
            return _worker_init_fn
    
        
        args = get_args()
        
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

        aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
            batch_size=mini_batch_size, train_folder = args.train_data_dir, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )

        sample_n_frames_bucket_interval = vae_model.vae.config.temporal_compression_ratio



        def collate_fn(examples):
            def get_length_to_frame_num(token_length):
                if args.image_sample_size > args.video_sample_size:
                    sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

                    if sample_sizes[-1] != args.image_sample_size:
                        sample_sizes.append(args.image_sample_size)
                else:
                    sample_sizes = [args.image_sample_size]
                
                length_to_frame_num = {
                    sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1 for sample_size in sample_sizes
                }


                return length_to_frame_num

            def get_random_downsample_ratio(sample_size, image_ratio=[],
                                            all_choices=False, rng=None):
                def _create_special_list(length):
                    if length == 1:
                        return [1.0]
                    if length >= 2:
                        first_element = 0.90
                        remaining_sum = 1.0 - first_element
                        other_elements_value = remaining_sum / (length - 1)
                        special_list = [first_element] + [other_elements_value] * (length - 1)
                        return special_list
                        
                if sample_size >= 1536:
                    number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
                elif sample_size >= 1024:
                    number_list = [1, 1.25, 1.5, 2] + image_ratio
                elif sample_size >= 768:
                    number_list = [1, 1.25, 1.5] + image_ratio
                elif sample_size >= 512:
                    number_list = [1] + image_ratio
                else:
                    number_list = [1]

                if all_choices:
                    return number_list

                number_list_prob = np.array(_create_special_list(len(number_list)))
                if rng is None:
                    return np.random.choice(number_list, p = number_list_prob)
                else:
                    return rng.choice(number_list, p = number_list_prob)

            # Get token length
            target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
            length_to_frame_num = get_length_to_frame_num(target_token_length)

            # Create new output
            new_examples                 = {}
            new_examples["target_token_length"] = target_token_length
            new_examples["pixel_values"] = []
            new_examples["text"]         = []
            # Used in Inpaint mode 
            if args.train_mode != "normal":
                new_examples["mask_pixel_values"] = []
                new_examples["mask"] = []
                new_examples["clip_pixel_values"] = []

            # Get downsample ratio in image and videos
            pixel_value     = examples[0]["pixel_values"]
            data_type       = examples[0]["data_type"]
            f, h, w, c      = np.shape(pixel_value)
            if data_type == 'image':
                random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])

                aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
                
                batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
            else:
                if args.random_hw_adapt:
                    if args.training_with_video_token_length:
                        local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["pixel_values"])[1], np.shape(example["pixel_values"])[2]])) for example in examples]))
                        # The video will be resized to a lower resolution than its own.
                        choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
                        if len(choice_list) == 0:
                            choice_list = list(length_to_frame_num.keys())
                        local_video_sample_size = np.random.choice(choice_list)
                        batch_video_length = length_to_frame_num[local_video_sample_size]
                        random_downsample_ratio = args.video_sample_size / local_video_sample_size
                    else:
                        random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
                        batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
                else:
                    random_downsample_ratio = 1
                    batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

                aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
                aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

            closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
            closest_size = [int(x / 16) * 16 for x in closest_size]
            if args.random_ratio_crop:
                random_sample_size = aspect_ratio_random_crop_sample_size[
                    np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
                ]
                random_sample_size = [int(x / 16) * 16 for x in random_sample_size]

            for example in examples:
                if args.random_ratio_crop:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    b, c, h, w = pixel_values.size()
                    th, tw = random_sample_size
                    if th / tw > h / w:
                        nh = int(th)
                        nw = int(w / h * nh)
                    else:
                        nw = int(tw)
                        nh = int(h / w * nw)
                    
                    transform = transforms.Compose([
                        transforms.Resize([nh, nw]),
                        transforms.CenterCrop([int(x) for x in random_sample_size]),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                else:
                    # To 0~1
                    pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
                    pixel_values = pixel_values / 255.

                    # Get adapt hw for resize
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / h > closest_size[1] / w:
                        resize_size = closest_size[0], int(w * closest_size[0] / h)
                    else:
                        resize_size = int(h * closest_size[1] / w), closest_size[1]
                    
                    transform = transforms.Compose([
                        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                        transforms.CenterCrop(closest_size),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                    ])
                new_examples["pixel_values"].append(transform(pixel_values))
                new_examples["text"].append(example["text"])

                batch_video_length = int(min(batch_video_length, len(pixel_values)))

                # Magvae needs the number of frames to be 4n + 1.
                batch_video_length = (batch_video_length - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1

                if batch_video_length <= 0:
                    batch_video_length = 1

                if args.train_mode != "normal":
                    mask = get_random_mask(new_examples["pixel_values"][-1].size())
                    mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) 
                    # Wan 2.1 use 0 for masked pixels
                    # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
                    new_examples["mask_pixel_values"].append(mask_pixel_values)
                    new_examples["mask"].append(mask)
                    
                    clip_pixel_values = new_examples["pixel_values"][-1][0].permute(1, 2, 0).contiguous()
                    clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
                    new_examples["clip_pixel_values"].append(clip_pixel_values)

            # Limit the number of frames to the same
            new_examples["pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["pixel_values"]])
            if args.train_mode != "normal":
                new_examples["mask_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["mask_pixel_values"]])
                new_examples["mask"] = torch.stack([example[:batch_video_length] for example in new_examples["mask"]])
                new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])

            # Encode prompts when enable_text_encoder_in_dataloader=True
            if args.enable_text_encoder_in_dataloader:
                prompt_ids = tokenizer(
                    new_examples['text'], 
                    max_length=args.tokenizer_max_length, 
                    padding="max_length", 
                    add_special_tokens=True, 
                    truncation=True, 
                    return_tensors="pt"
                )
                encoder_hidden_states = text_encoder(
                    prompt_ids.input_ids
                )[0]
                new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
                new_examples['encoder_hidden_states'] = encoder_hidden_states

            return new_examples
        
        # DataLoaders creation:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            # persistent_workers=True if args.dataloader_num_workers != 0 else False,
            # num_workers=args.dataloader_num_workers,
            # worker_init_fn=worker_init_fn(args.seed + accelerator.process_index)
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
