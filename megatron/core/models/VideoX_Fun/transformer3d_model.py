# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from collections import OrderedDict
from typing import Dict, Literal, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.cuda.amp as amp
import math
from megatron.core import parallel_state
from megatron.core import tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import (
    MultimodalRotaryEmbedding,
    RotaryEmbedding,
)
from diffusers.training_utils import compute_loss_weighting_for_sd3
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    tie_output_layer_state_dict,
    tie_word_embeddings_state_dict,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock, Transformer3DBlock
from megatron.core.transformer.transformer_config import TransformerConfig, WanTransformerConfig
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from megatron.core.transformer.torch_norm import WanRMSNorm, WanLayerNorm
import torch.nn.functional as F


def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs



class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens




class Head(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.out_dim = config.out_dim
        self.patch_size = config.patch_size

        # layers
        out_dim = math.prod(self.patch_size) * self.out_dim
        self.norm = WanLayerNorm(config, config.hidden_size)
        self.head = nn.Linear(self.hidden_size, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, config.hidden_size) / config.hidden_size**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        e = tuple(t.squeeze(1) for t in e)
        x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    diff = noise_pred - target[:, :noise_pred.size(1), :]
    mse_loss = F.mse_loss(noise_pred, target[:, :noise_pred.size(1), :], reduction='none')
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    return final_loss


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x



class WanTransformer3DModel(LanguageModule):
    """GPT Transformer language model.

    Args:
        config (TransformerConfig):
            Transformer config
        transformer_layer_spec (ModuleSpec):
            Specifies module to use for transformer layers
        vocab_size (int):
            Vocabulary size
        max_sequence_length (int):
            maximum size of sequence. This is used for positional embedding
        pre_process (bool, optional):
            Include embedding layer (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional):
            Include an output layer (used with pipeline parallelism). Defaults to True.
        fp16_lm_cross_entropy (bool, optional):
            Defaults to False.
        parallel_output (bool, optional):
            Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional):
            When True, input embeddings and output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope], optional):
            Position embedding type.. Defaults to 'learned_absolute'.
        rotary_percent (float, optional):
            Percent of rotary dimension to use for rotary position embeddings.
            Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional):
            Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'.
            Defaults to 10000.
        rope_scaling (bool, optional): Toggle RoPE scaling.
        rope_scaling_factor (float): RoPE scaling factor. Default 8.
        scatter_embedding_sequence_parallel (bool, optional):
            Whether embeddings should be scattered across sequence parallel
            region or not. Defaults to True.
        seq_len_interpolation_factor (Optional[float], optional):
            scale of linearly interpolating RoPE for longer sequences.
            The value must be a float larger than 1.0. Defaults to None.
    """

    def __init__(
        self,
        config: WanTransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        self.model_type = config.model_type
        self.patch_size = config.patch_size
        self.text_len = config.text_len
        self.in_dim = config.in_dim
        self.freq_dim = config.freq_dim
        self.text_dim = config.text_dim
        self.out_dim = config.out_dim
        self.window_size = config.window_size
        self.qk_norm = config.qk_norm
        self.cross_attn_norm = config.cross_attn_norm
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.text_dim = config.text_dim


        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.vp_stage = vp_stage

        
        self.patch_embedding = nn.Conv3d(
            self.in_dim, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden_size), nn.GELU(approximate='tanh'),
            nn.Linear(self.hidden_size, self.hidden_size))
        
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hidden_size), nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_size, self.hidden_size * 6))


        if hasattr(self.config, 'position_embedding_type'):
            self.position_embedding_type = self.config.position_embedding_type
        else:
            self.position_embedding_type = position_embedding_type

        # # megatron core pipelining currently depends on model type
        # # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        # # These 4 attributes are needed for TensorRT-LLM export.


        head_hidden_size = self.hidden_size // self.num_attention_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, head_hidden_size - 4 * (head_hidden_size // 6)),
                rope_params(1024, 2 * (head_hidden_size // 6)),
                rope_params(1024, 2 * (head_hidden_size // 6))
            ],
            dim=1
        )

        self.img_emb = MLPProj(1280, self.hidden_size)

       
        self.decoder = Transformer3DBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
            vp_stage=vp_stage,
        )

        # self.head = Head(config.hidden_size, config.out_dim, confipatch_size, eps)
        self.head = Head(config)



    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        # assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor)


    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        grid_sizes = grid_sizes.to(torch.int64)
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def split_cp_sp(self, x):
        # CP 拆分
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        seq_len = x.size(0)
        assert seq_len % cp_size == 0, "seq_len must be divisible by cp_size"
        cp_intervel = seq_len // cp_size
        x = x[(cp_rank * cp_intervel):(cp_rank + 1) * cp_intervel, :, :]

        # TP 拆分
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        seq_len = x.size(0)
        assert seq_len % tp_size == 0, "hidden size must be divisible by tp_size"
        tp_interval = seq_len // tp_size
        assert tp_interval % 2 == 0
        x = x[(tp_rank * tp_interval):(tp_rank + 1) * tp_interval, :, :]

        return x

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor,
        grid_sizes: Tensor,
        context_mask: Tensor,
        attention_mask: Tensor,
        clip_fea: Tensor = None,
        y: Tensor = None,
        sigmas: Tensor = None,
        target: Tensor = None,
        y_camera: Tensor = None,
        full_ref: Tensor = None,
        cond_flag: bool = True,
        packed_seq_params: Optional[PackedSeqParams] = None,
    ) -> Tensor:
        """Forward function of the GPT Model This function passes the input tensors
        through the embedding layer, and then the decoeder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given  or the final hidden units

        Args:
            runtime_gather_output (bool): Gather output at runtime. Default None means
                `parallel_output` arg in the constructor will be used.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        if parallel_state.is_pipeline_first_stage():
    
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            x = [self.patch_embedding(u.unsqueeze(0)) for u in x] 
            
            x = [u.flatten(2).transpose(1, 2) for u in x]
            
            pad_seq_len = int(grid_sizes[0][0]) * int(grid_sizes[0][1]) * int(grid_sizes[0][2])
            x = torch.cat([
                torch.cat([u, u.new_zeros(1, pad_seq_len - u.size(1), u.size(2))],
                          dim=1) for u in x # padding
            ])
            
            context_lens = None
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context # padding 
                ]))
    
            if clip_fea is not None:
                context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
                context = torch.concat([context_clip, context], dim=1)

            x = x.transpose(0, 1).contiguous()
            context = context.transpose(0, 1).contiguous()
            x = self.split_cp_sp(x)

            print("x shape", x.shape)

            
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float())#.to(torch.bfloat16)
            e0 = self.time_projection(e).unflatten(1, (6, self.hidden_size)) # .to(torch.bfloat16)


        if self.freqs.device == torch.device('cpu'):
            device = e.device
            assert device.type == 'cuda', f"Expected CUDA device, got {device}"
            self.freqs = self.freqs.cuda()


        x = self.decoder(
            hidden_states=x,
            e = e0,
            grid_sizes = grid_sizes,
            attention_mask=attention_mask,
            freqs=self.freqs,
            context=context,
            context_mask=context_mask,
            packed_seq_params=packed_seq_params,
        )

        
        if parallel_state.is_pipeline_last_stage():
            x = self.head(x, e)
            x = x.transpose(0, 1).contiguous()
            print(f"megatron x : {x} {x.shape}")
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=None, sigmas=sigmas)
            loss = custom_mse_loss(x.to(torch.float32), target.to(torch.float32), weighting.to(torch.float32))
            return loss
        else:
            raise NotImplementedError("don't support pp now!")
            grid_sizes = grid_sizes.reshape(1, grid_sizes.size(0), grid_sizes.size(1))
            return [x, context, target, grid_sizes]


    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility.

        Removing extra state.
        Tie word embeddings and output layer in mtp process stage.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        output_layer_extra_state_key = f'{prefix}output_layer._extra_state'

        # Old GPT checkpoints only stored the output layer weight key. So we remove the
        # _extra_state key but check that it doesn't contain any data anyway
        output_extra_state = sharded_state_dict.pop(output_layer_extra_state_key, None)
        assert not (
            output_extra_state and output_extra_state.data
        ), f'Expected output layer extra state to be empty, got: {output_extra_state}'

        # Multi-Token Prediction (MTP) need both embedding layer and output layer in
        # mtp process stage.
        # If MTP is not placed in the pre processing stage, we need to maintain a copy of
        # embedding layer in the mtp process stage and tie it to the embedding in the pre
        # processing stage.
        # Also, if MTP is not placed in the post processing stage, we need to maintain a copy
        # of output layer in the mtp process stage and tie it to the output layer in the post
        # processing stage.
        if self.mtp_process and not self.pre_process:
            emb_weight_key = f'{prefix}embedding.word_embeddings.weight'
            emb_weight = self.embedding.word_embeddings.weight
            tie_word_embeddings_state_dict(sharded_state_dict, emb_weight, emb_weight_key)
        if self.mtp_process and not self.post_process:
            # We only need to tie the output layer weight if share_embeddings_and_output_weights
            # is False. Because if share_embeddings_and_output_weights is True, the shared weight
            # will be stored in embedding layer, and output layer will not have any weight.
            if not self.share_embeddings_and_output_weights:
                output_layer_weight_key = f'{prefix}output_layer.weight'
                output_layer_weight = self.output_layer.weight
                tie_output_layer_state_dict(
                    sharded_state_dict, output_layer_weight, output_layer_weight_key
                )

        return sharded_state_dict
