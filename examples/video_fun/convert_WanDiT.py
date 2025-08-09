# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain GPT."""
import sys
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import random
from tqdm import tqdm
import time

from functools import partial
from typing import List, Optional, Tuple, Union
from megatron.core import parallel_state
from megatron.training import inprocess_restart
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.dist_checkpointing import load, load_plain_tensors, save
from megatron.core import dist_checkpointing

from megatron.core.transformer.module import Float16Module

from megatron.core.models.VideoX_Fun.transformer3d_layer_specs import (
    get_transformer3d_layer_local_spec,
    get_transformer3d_transformer_engine_block_spec
)

from megatron.core.models.T5.t5_spec import (
    get_t5_encoder_with_transformer_engine_block_spec,
    get_t5_encoder_with_local_block_spec,
)

import  megatron.core.models.VideoX_Fun as VF
from megatron.core.models.T5 import T5Model
from megatron.core.models.wan.wan_vae import WanVae, AutoencoderKLWan
from megatron.core.models.wan.wan_image_encoder import WanCLIP, get_wan_clip_spec, get_wan_clip_spec_te

# from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
#     get_gpt_heterogeneous_layer_spec,
# )
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer3d_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.arguments import parse_args, validate_args
from megatron.training.async_utils import init_persistent_async_worker
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.arguments import core_transformer3d_config_from_args
from megatron.core.dist_checkpointing.validation import StrictHandling

import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

from videox_fun.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                               WanTransformer3DModel)

from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)

stimer = StragglerDetector()

def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    diff = noise_pred - target[:, :noise_pred.size(1), :]
    mse_loss = F.mse_loss(noise_pred, target[:, :noise_pred.size(1), :], reduction='none')
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    return final_loss

def set_seed(seed):
    """
    设置所有随机源的种子以确保可复现性
    
    参数:
        seed (int): 随机种子值
    """
    # 1. Python内置随机模块
    random.seed(seed)
    
    # 2. Numpy随机生成器
    np.random.seed(seed)
    
    # 3. PyTorch CPU随机种子
    torch.manual_seed(seed)
    
    # 4. PyTorch GPU随机种子（所有GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
    
    # 5. 设置CuDNN以保证确定性（可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    # Torch setup for distributed training
    args = parse_args()
    args_defaults={'tokenizer_type': 'GPT2BPETokenizer'}
    # Prep for checkpoint conversion.
    if args.ckpt_convert_format is not None:
        assert args.ckpt_convert_save is not None
        assert args.load is not None
        args.exit_on_missing_checkpoint = True

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoint-args requires --load argument"
        assert args.non_persistent_ckpt_type != "local", (
            "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
            "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
            "before initializing LocalCheckpointManager."
        )
        load_args_from_checkpoint(args)

    if args.async_save and args.use_persistent_ckpt_worker:
        init_persistent_async_worker()

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)
        
    set_global_variables(args)
    
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)
    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(args.tensor_model_parallel_size, args.pipeline_model_parallel_size)
    

def set_preprocess_state(args, model, hf_model):
    model.patch_embedding.weight.data.copy_(hf_model.patch_embedding.weight.data)
    model.patch_embedding.bias.data.copy_(hf_model.patch_embedding.bias.data)
    model.text_embedding[0].weight.data.copy_(hf_model.text_embedding[0].weight.data)
    model.text_embedding[0].bias.data.copy_(hf_model.text_embedding[0].bias.data)
    model.text_embedding[2].weight.data.copy_(hf_model.text_embedding[2].weight.data)
    model.text_embedding[2].bias.data.copy_(hf_model.text_embedding[2].bias.data)
    model.time_embedding[0].weight.data.copy_(hf_model.time_embedding[0].weight.data)
    model.time_embedding[0].bias.data.copy_(hf_model.time_embedding[0].bias.data)
    model.time_embedding[2].weight.data.copy_(hf_model.time_embedding[2].weight.data)
    model.time_embedding[2].bias.data.copy_(hf_model.time_embedding[2].bias.data)
    model.time_projection[1].weight.data.copy_(hf_model.time_projection[1].weight.data)
    model.time_projection[1].bias.data.copy_(hf_model.time_projection[1].bias.data)


def set_postprocess_state(args, model, hf_model):
    model.head.modulation.data.copy_(hf_model.head.modulation.data)
    model.head.head.weight.data.copy_(hf_model.head.head.weight.data)
    model.head.head.bias.data.copy_(hf_model.head.head.bias.data)
    model.img_emb.proj[0].weight.data.copy_(hf_model.img_emb.proj[0].weight.data)
    model.img_emb.proj[0].bias.data.copy_(hf_model.img_emb.proj[0].bias.data)
    model.img_emb.proj[1].weight.data.copy_(hf_model.img_emb.proj[1].weight.data)
    model.img_emb.proj[1].bias.data.copy_(hf_model.img_emb.proj[1].bias.data)
    model.img_emb.proj[3].weight.data.copy_(hf_model.img_emb.proj[3].weight.data)
    model.img_emb.proj[3].bias.data.copy_(hf_model.img_emb.proj[3].bias.data)  
    model.img_emb.proj[4].weight.data.copy_(hf_model.img_emb.proj[4].weight.data)
    model.img_emb.proj[4].bias.data.copy_(hf_model.img_emb.proj[4].bias.data)  
    return

def set_selfattn_state(args, layer, hf_layer):
    attn = layer.self_attention
    hf_attn = hf_layer.self_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size 
    nh = args.num_attention_heads // tp
    ng = args.num_attention_heads // tp
    dim = args.kv_channels
    assert nh % ng == 0


    attn.linear_qkv.weight.data.copy_(torch.cat([
        hf_attn.q.weight.reshape((ng, dim*nh//ng, -1)),
        hf_attn.k.weight.reshape((ng, dim, -1)),
        hf_attn.v.weight.reshape((ng, dim, -1)),
    ], dim=1).reshape((-1, args.hidden_size)))

    
    attn.linear_qkv.bias.data.copy_(torch.cat([
            hf_attn.q.bias.reshape((ng, dim*nh//ng)),
            hf_attn.k.bias.reshape((ng, dim)),
            hf_attn.v.bias.reshape((ng, dim)),
        ], dim=1).reshape(-1))
    
    attn.linear_proj.weight.data.copy_(hf_attn.o.weight.data)
    attn.linear_proj.bias.data.copy_(hf_attn.o.bias.data)
    
    attn.q_layernorm.weight.data.copy_(hf_attn.norm_q.weight.data)
    attn.k_layernorm.weight.data.copy_(hf_attn.norm_k.weight.data)
    
    return

def set_corssattn_state(args, layer, hf_layer):
    attn = layer.cross_attention
    hf_attn = hf_layer.cross_attn

    # Reshape loaded weights.
    tp = args.tensor_model_parallel_size 
    nh = args.num_attention_heads // tp
    ng = args.num_attention_heads // tp
    dim = args.kv_channels
    assert nh % ng == 0
    
    attn.linear_q.weight.data.copy_(hf_attn.q.weight.data)
    attn.linear_q.bias.data.copy_(hf_attn.q.bias.data)
    
    attn.linear_kv.weight.data.copy_(torch.cat([hf_attn.k.weight.reshape((ng, dim, -1)),
                                                hf_attn.v.weight.reshape((ng, dim, -1))], dim=1).reshape((-1, args.hidden_size)))
    attn.linear_kv.bias.data.copy_(torch.cat([hf_attn.k.bias.reshape((ng, dim)), 
                                              hf_attn.v.bias.reshape((ng, dim))], dim=1).reshape(-1))
    
    attn.linear_proj.weight.data.copy_(hf_attn.o.weight.data)
    attn.linear_proj.bias.data.copy_(hf_attn.o.bias.data)
    
    attn.linear_kv_img.weight.data.copy_(torch.cat([hf_attn.k_img.weight.reshape((ng, dim, -1)), 
                                                    hf_attn.v_img.weight.reshape((ng, dim, -1))], dim=1).reshape((-1, args.hidden_size)))
    attn.linear_kv_img.bias.data.copy_(torch.cat([hf_attn.k_img.bias.reshape((ng, dim)),
                                                  hf_attn.v_img.bias.reshape((ng, dim))], dim=1).reshape(-1))
    
    attn.q_layernorm.weight.data.copy_(hf_attn.norm_q.weight.data)
    attn.k_layernorm.weight.data.copy_(hf_attn.norm_k.weight.data)
    attn.k_layernorm_img.weight.data.copy_(hf_attn.norm_k_img.weight.data)
    
    
    

def set_mlp_state(args, layer, hf_layer):
    mlp = layer.mlp
    hf_mlp = hf_layer.ffn
    mlp.linear_fc1.weight.data.copy_(hf_mlp[0].weight.data)
    mlp.linear_fc1.bias.data.copy_(hf_mlp[0].bias.data)
    mlp.linear_fc2.weight.data.copy_(hf_mlp[2].weight.data)
    mlp.linear_fc2.bias.data.copy_(hf_mlp[2].bias.data)



def set_layer_state(args, model, hf_model, layer_idx):
    layer = model.decoder.layers[layer_idx]
    hf_layer = hf_model.blocks[layer_idx]
    layer.modulation.data.copy_(hf_layer.modulation.data)
    
    set_selfattn_state(args, layer, hf_layer)
    set_corssattn_state(args, layer, hf_layer)
    set_mlp_state(args, layer, hf_layer)

    layer.pre_cross_attn_layernorm.weight.data.copy_(hf_layer.norm3.weight.data)
    layer.pre_cross_attn_layernorm.bias.data.copy_(hf_layer.norm3.bias.data)

def save_distributed_checkpoint(wanclip_model, checkpoint_path):
    sharded_state_dict = wanclip_model.sharded_state_dict(prefix='')
    # print(sharded_state_dict)
    dist_checkpointing.save(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)

def load_distributed_checkpoint(gpt_model, checkpoint_path):
    sharded_state_dict=gpt_model.sharded_state_dict(prefix='')
    checkpoint = dist_checkpointing.load(sharded_state_dict=sharded_state_dict, checkpoint_dir=checkpoint_path)
    gpt_model.load_state_dict(checkpoint)
    return gpt_model

def convert_DiT(dit_path, save_path = None):
    vae_fea = torch.randn(1, 16, 6, 90, 156, dtype=torch.bfloat16).cuda()
    t = torch.tensor([0]).cuda()  # 假设 t 是时间步长标记之类的
    context = [torch.randn(209, 4096, dtype=torch.bfloat16).cuda()]
    seqlen = 21060  # 这是个整数，不用放 device 上
    inpaint_latents = torch.randn(1, 20, 6, 90, 156, dtype=torch.bfloat16).cuda()
    clip_context = torch.randn(1, 257, 1280, dtype=torch.bfloat16).cuda()
    hidden_state_seq_len = 21060
    context_seqlen = 769
    grid_sizes = torch.tensor([[ 6, 45, 78]], dtype=torch.int32).cuda()
    sigmas = torch.tensor([0.1], dtype=torch.bfloat16).reshape(1,1,1,1,1).cuda()
    target = torch.randn(1, 21060, 64, dtype=torch.bfloat16).cuda()
    
    
    args = get_args()
    pre_process = True
    post_process = True
    config = core_transformer3d_config_from_args(args)
    config.text_dim = 4096
    config.layernorm_epsilon = 1e-6
    config.attention_dropout=0.0
    transformer_layer_spec = get_transformer3d_transformer_engine_block_spec()
    vp_stage = None

    additional_kwargs = {
        'transformer_subpath': './',
        'dict_mapping': {
            'in_dim': 'in_channels',
            'dim': 'hidden_size'
        }
    }
    hf_dit = WanTransformer3DModel.from_pretrained(
        dit_path,
        transformer_additional_kwargs=additional_kwargs,
    )
    if torch.distributed.get_rank() == 0:
        print(f"hf dit : {hf_dit}")
    
    
    megatron_dit = VF.WanTransformer3DModel(
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
    if torch.distributed.get_rank() == 0:
        print(f"megatron dit : {megatron_dit}")
    

    
    set_preprocess_state(args, megatron_dit, hf_dit)
    set_postprocess_state(args, megatron_dit, hf_dit)
    for layer_idx in tqdm(range(len(megatron_dit.decoder.layers))):
        set_layer_state(config, megatron_dit, hf_dit, layer_idx)

    if save_path:
        save_distributed_checkpoint(megatron_dit, save_path)
    
    megatron_dit.to(torch.bfloat16).cuda()
    hf_dit.to(torch.bfloat16).cuda()
    megatron_dit.requires_grad_(False)
    hf_dit.requires_grad_(False)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        hf_output = hf_dit(x=vae_fea,
            context=context,
            t=t,
            seq_len=hidden_state_seq_len,
            y=inpaint_latents,
            clip_fea=clip_context,)
        megatron_output = megatron_dit(vae_fea, t, context, grid_sizes, None, None, clip_context, inpaint_latents, sigmas, target)
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=None, sigmas=sigmas)
        loss = custom_mse_loss(hf_output.to(torch.float32), target.reshape(1, 16, 6, 90, 156).to(torch.float32), weighting.to(torch.float32))


    if torch.distributed.get_rank() == 0:
        print(f"megatron_output {megatron_output} {megatron_output.shape}")
        print(f"hf_output {loss} {loss.shape}")
    
    return



if __name__ == "__main__":
    seed = 43
    set_seed(seed)
    initialize_distributed()
    model_parallel_cuda_manual_seed(seed)
    dit_path = "path/to/Wan2.1-Fun-V1.1-14B-InP" # 原模型checkpoint的位置
    save_path = "path/to/dit" # megatron 格式的checkpoint的存储位置
    convert_DiT(dit_path)
