# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Optional, Union
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.models.common.vision_module.vision_module import VisionModule
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from diffusers.models.modeling_utils import ModelMixin


from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TEColumnParallelLinear,
    TERowParallelLinear,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import TENorm

    NORM_IMPL = TENorm
except:
    NORM_IMPL = torch.nn.LayerNorm


@dataclass
class WanCLIPSubmodules:
    WanCLIPVisual: Union[ModuleSpec, type] = None
    WanCLIPTextual: Union[ModuleSpec, type] = None
    
@dataclass
class WanCLIPViTSubmodules:
    WanCLIPViTAttn: Union[ModuleSpec, type] = None

@dataclass
class WanCLIPXLMSubModules:
    WanCLIPXLMAttn: Union[ModuleSpec, type] = None


@dataclass
class WanCLIPViTAttnBlockSubmodules:
    self_attention : Union[ModuleSpec, type] = None
    mlp: Union[ModuleSpec, type] = None

    
    
def get_wan_clip_spec():
    
    visual_attn_block = WanCLIPViTAttnBlockSubmodules(
        self_attention = ModuleSpec(
            module = SelfAttention,
            submodules = SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ) 
        ),
        mlp = ModuleSpec(
            module = MLP,
            submodules = MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ),
        )
    )
    visual_attn_spec = ModuleSpec(
        module = WanCLIPViTAttentionBlock,
        submodules = visual_attn_block,
    )
    

    return visual_attn_spec

def get_wan_clip_spec_te():
    
    visual_attn_block = WanCLIPViTAttnBlockSubmodules(
        self_attention = ModuleSpec(
            module = SelfAttention,
            submodules = SelfAttentionSubmodules(
                linear_qkv=TEColumnParallelLinear,
                core_attention=TEDotProductAttention,
                linear_proj=TERowParallelLinear,
                q_layernorm=IdentityOp,
                k_layernorm=IdentityOp,
            ) 
        ),
        mlp = ModuleSpec(
            module = MLP,
            submodules = MLPSubmodules(
                linear_fc1=TEColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ),
        )
    )
    visual_attn_spec = ModuleSpec(
        module = WanCLIPViTAttentionBlock,
        submodules = visual_attn_block,
    )
    

    return visual_attn_spec


def _get_WanCLIP_submodules(
    config: TransformerConfig,
    spec: Union[WanCLIPSubmodules, ModuleSpec],
    vp_stage: Optional[int] = None,
) -> WanCLIPSubmodules:
    """
    Retrieve or construct TransformerBlockSubmodules based on the provided specification.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
        spec (Union[TransformerBlockSubmodules, ModuleSpec]): Specification for the
            transformer block submodules. Can be either a TransformerBlockSubmodules
            instance or a ModuleSpec.
        vp_stage (Optional[int]): Virtual pipeline stage number.

    Returns:
        TransformerBlockSubmodules: The submodules for the transformer block.
    """

    # Transformer block submodules.
    if isinstance(spec, WanCLIPSubmodules):
        return spec

    # ModuleSpec here is generally assumed to be for a transformer layer that
    # is implemented in `transformer_layer.py` or if it subclasses
    # `BaseTransformerLayer` from the `transformer_layer.py` file.
    elif isinstance(spec, ModuleSpec):
        if issubclass(spec.module, WanCLIPSubmodules):
            return spec.submodules
        elif issubclass(spec.module, ):
            num_layers = get_num_layers_to_build(config, vp_stage)
            return TransformerBlockSubmodules(
                layer_specs=[spec] * num_layers, layer_norm=LayerNormImpl
            )
        else:
            raise Exception(f"specialize for {spec.module.__name__}.")
    else:
        raise Exception(f"specialize for {type(spec).__name__}.")


    
def pos_interpolate(pos, seq_len):
    if pos.size(1) == seq_len:
        return pos
    else:
        src_grid = int(math.sqrt(pos.size(1)))
        tar_grid = int(math.sqrt(seq_len))
        n = pos.size(1) - src_grid * src_grid
        return torch.cat([
            pos[:, :n],
            F.interpolate(
                pos[:, n:].float().reshape(1, src_grid, src_grid, -1).permute(
                    0, 3, 1, 2),
                size=(tar_grid, tar_grid),
                mode='bicubic',
                align_corners=False).flatten(2).transpose(1, 2)
        ],
                         dim=1)

class QuickGELU(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)




class WanCLIPViTAttentionBlock(VisionModule):
    def __init__(self,
                 config: TransformerConfig,
                 wan_clip_vit_attn_block_spec: Union[ModuleSpec, WanCLIPViTAttnBlockSubmodules],
                 dim: int,
                 post_norm : bool = False,
                 causal: bool = False,
                 activation: str = 'quick_gelu',
                 norm_eps: int = 1e-5,
                 ):
        super().__init__(config)
        self.dim = dim
        self.post_norm = post_norm
        self.causal = causal
        if self.causal:
            self.attn_mask = None # 根据序列长度确定
        else:
            self.attn_mask = None
        self.norm_eps = norm_eps
        self.norm1 = build_module(
            NORM_IMPL,
            config=config,
            hidden_size=self.dim,
            eps=self.norm_eps)
        
        self.attn = SelfAttention(
            config,
            submodules = wan_clip_vit_attn_block_spec.self_attention.submodules,
            layer_number=1,
        )
        self.norm2 = build_module(
            NORM_IMPL,
            config=config,
            hidden_size=self.dim,
            eps=self.norm_eps,)

        self.mlp = MLP(
            config = config,
            submodules = wan_clip_vit_attn_block_spec.mlp.submodules,
            input_size=self.dim,
        )

    def forward(self, x, attention_mask):
        if self.post_norm:
            x = x + self.norm1(self.attn(x, attention_mask)[0])
            x = x + self.norm2(self.mlp(x)[0])
        else:
            x = x + self.attn(self.norm1(x), attention_mask)[0]
            x = x + self.mlp(self.norm2(x))[0]
        return x



class WanCLIPVisionTransformer(VisionModule):
    """Vision Transformer model.

    Args:
        transformer_config (TransformerConfig): Transformer config.
        transformer_layer_spec (ModuleSpec): Specifies module to use for transformer layers.
        ln_pre_impl (ModuleSpec or type): Specifies the layer norm type to use for ln_pre.
        add_class_token (bool, optional): Include a class token. Defaults to True.
        class_token_len (int): Class token length. Defaults to 1 but 8 may be faster.
        patch_dim (int): Image patch size.
        img_h (int): Input image height.
        img_w (int): Input image width.
    """
    def __init__(
        self,
        transformer_config: TransformerConfig,
        wan_clip_vit_spec: Union[ModuleSpec, WanCLIPViTSubmodules],
        image_size: int = 224,
        patch_size: int = 14,
        dim: int = 1280,
        mlp_ratio: int = 4,
        out_dim: int = 1024,
        num_heads: int = 16,
        num_layers: int = 32,
        pool_type: str = 'token',
        pre_norm: bool = True,
        post_norm: bool = False,
        activation: str = 'quick_gelu',
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        
        if image_size % patch_size != 0:
            print(
                '[WARNING] image_size is not divisible by patch_size',
                flush=True)
        assert pool_type in ('token', 'token_fc', 'attn_pool')
        out_dim = out_dim or dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size)**2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.post_norm = post_norm
        self.norm_eps = norm_eps
        self.activation = activation
        
        # transformer_config.num_layers = num_layers
        # transformer_config.hidden_size = dim
        # transformer_config.ffn_hidden_size = dim * mlp_ratio
        # transformer_config.num_attention_heads = num_heads
        # transformer_config.attention_dropout = attn_dropout
        # transformer_config.hidden_dropout = proj_dropout
        # transformer_config.layernorm_epsilon = norm_eps
        # transformer_config.apply_residual_connection_post_layernorm = post_norm
        
        super().__init__(config=transformer_config)
        
        self.patch_embedding = nn.Conv2d(
            in_channels = 3,
            out_channels = dim,
            kernel_size = patch_size,
            stride = patch_size,
            bias = not pre_norm,
        )
        gain = 1.0 / math.sqrt(dim)
        if pool_type in ('token', 'token_fc'):
            self.cls_embedding = nn.Parameter(gain * torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(gain * torch.randn(
            1, self.num_patches +
            (1 if pool_type in ('token', 'token_fc') else 0), dim))
        self.dropout = nn.Dropout(embedding_dropout)
        
        
        self.pre_norm = build_module(
            NORM_IMPL,
            config=transformer_config,
            hidden_size=dim,
            eps=norm_eps)
        # transformer layers        
        self.transformer = nn.ModuleList([
            WanCLIPViTAttentionBlock(
                config=transformer_config,
                wan_clip_vit_attn_block_spec=wan_clip_vit_spec.submodules,
                dim = self.dim,
                post_norm = self.post_norm,
                causal = False,
                activation = self.activation,
                norm_eps = self.norm_eps,
            ) for _ in range(self.num_layers)
        ])
        
        self.post_norm = build_module(
            NORM_IMPL,
            config=transformer_config,
            hidden_size=dim,
            eps=norm_eps,
        )
        
        # head
        if pool_type == 'token':
            self.head = nn.Parameter(gain * torch.randn(dim, out_dim))
        elif pool_type == 'token_fc':
            self.head = nn.Linear(dim, out_dim)
        elif pool_type == 'attn_pool':
            self.head = AttentionPool(dim, mlp_ratio, num_heads, activation,
                                      proj_dropout, norm_eps)
    
    def forward(self, x, attention_mask, interpolation=False, use_31_block=False):
        b = x.size(0)

        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)

        if self.pool_type in ('token', 'token_fc'):
            x = torch.cat([self.cls_embedding.expand(b, -1, -1), x], dim=1)
        if interpolation:
            e = pos_interpolate(self.pos_embedding, x.size(1))
        else:
            e = self.pos_embedding

        x = self.dropout(x + e)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer
        if use_31_block:
            for layer in self.transformer[:-1]:
                x = layer(x, attention_mask)
            return x
        else:
            for layer in self.transformer:
                x = layer(x, attention_mask)
            return x


# class XLMRobertaWithHead(XLMRoberta):
#     def __init__(self, 
#                  config: TransformerConfig,
#                  wan_clip_xlm_spec: ModuleSpec, 
#                  **kwargs):
#         self.out_dim = kwargs.pop('out_dim')
#         super().__init__(config)
        
#         mid_dim = (self.dim + self.out_dim) // 2
#         """
#         self.head = nn.Sequential(
#             nn.Linear(self.dim, mid_dim, bias=False), nn.GELU(),
#             nn.Linear(mid_dim, self.out_dim, bias=False))
#         """
#         self.head = nn.Sequential(
#                                 ColumnParallelLinear(self.dim, mid_dim, gather_output=True, bias=False),
#                                 nn.GELU(),
#                                 RowParallelLinear(mid_dim, self.out_dim, input_is_parallel=True, bias=False)
#                                 )
        
#     def forward(self, ids):
#         x = super().forward(ids)

#         # average pooling
#         mask = ids.ne(self.pad_id).unsqueeze(-1).to(x)
#         x = (x * mask).sum(dim=1) / mask.sum(dim=1)

#         x = self.head(x)        
#         return x


class XLMRobertaCLIP(VisionModule):
    def __init__(self,
                config: TransformerConfig,
                wan_clip_attn_spec: ModuleSpec,
                embed_dim=1024,
                image_size=224,
                patch_size=14,
                vision_dim=1280,
                vision_mlp_ratio=4,
                vision_heads=16,
                vision_layers=32,
                vision_pool='token',
                vision_pre_norm=True,
                vision_post_norm=False,
                activation='gelu',
                vocab_size=250002,
                max_text_len=514,
                type_size=1,
                pad_id=1,
                text_dim=1024,
                text_heads=16,
                text_layers=24,
                text_post_norm=True,
                text_dropout=0.1,
                attn_dropout=0.0,
                proj_dropout=0.0,
                embedding_dropout=0.0,
                norm_eps=1e-5):
        super().__init__(config)
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_post_norm = text_post_norm
        self.norm_eps = norm_eps
        
        # models
        self.visual = WanCLIPVisionTransformer(
            transformer_config=config,
            wan_clip_vit_spec=wan_clip_attn_spec,
            image_size=image_size,  
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps)
        
        # self.textual = XLMRobertaWithHead(
        #     config=config,
        #     wan_clip_xlm_spec=wan_clip_attn_spec,
        #     vocab_size=vocab_size,
        #     max_seq_len=max_text_len,
        #     type_size=type_size,
        #     pad_id=pad_id,
        #     dim=text_dim,
        #     out_dim=embed_dim,
        #     num_heads=text_heads,
        #     num_layers=text_layers,
        #     post_norm=text_post_norm,
        #     dropout=text_dropout)
        self.log_scale = nn.Parameter(math.log(1 / 0.07) * torch.ones([]))
        
    def forward(self, imgs, txt_ids):
        """
        imgs:       [B, 3, H, W] of torch.float32.
        - mean:     [0.48145466, 0.4578275, 0.40821073]
        - std:      [0.26862954, 0.26130258, 0.27577711]
        txt_ids:    [B, L] of torch.long.
                    Encoded by data.CLIPTokenizer.
        """
        xi = self.visual(imgs)
        #xt = self.textual(txt_ids)
        return xi # , xt

    def param_groups(self):
        groups = [{
            'params': [
                p for n, p in self.named_parameters()
                if 'norm' in n or n.endswith('bias')
            ],
            'weight_decay': 0.0
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if not ('norm' in n or n.endswith('bias'))
            ]
        }]
        return groups

def _clip(config: TransformerConfig,
          wan_clip_attn_spec: ModuleSpec,
          pretrained=False,
          pretrained_name=None,
          model_cls=XLMRobertaCLIP,
          return_transforms=False,
          return_tokenizer=False,
          tokenizer_padding='eos',
          dtype=torch.float32,
          device='cpu',
          **kwargs):
    # init a model on device
    with torch.device(device):
        model = model_cls(config = config, wan_clip_attn_spec = wan_clip_attn_spec,**kwargs)

    # set device
    model = model.to(dtype=dtype, device=device)
    output = (model,)

    # init transforms
    if return_transforms:
        # mean and std
        if 'siglip' in pretrained_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]

        # transforms
        transforms = T.Compose([
            T.Resize((model.image_size, model.image_size),
                     interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])
        output += (transforms,)
    return output[0] if len(output) == 1 else output

def clip_xlm_roberta_vit_h_14(
        config : TransformerConfig,
        wan_clip_attn_spec: ModuleSpec,
        pretrained=False,
        pretrained_name='open-clip-xlm-roberta-large-vit-huge-14',
        **kwargs):
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool='token',
        activation='gelu',
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        text_post_norm=True,
        text_dropout=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0)
    cfg.update(**kwargs)
    return _clip(config, wan_clip_attn_spec,  pretrained, pretrained_name, XLMRobertaCLIP, **cfg)

class WanCLIP(VisionModule, ModelMixin):
    """Wan CLIP model.

    Args:
        wan_clip_config (TransformerConfig): Transformer config.
        wan_clip_spec (ModuleSpec): Specifies module to use for transformer layers.
    """
    def __init__(
        self,
        wan_clip_config: TransformerConfig,
        wan_clip_attn_spec: Union[WanCLIPSubmodules, ModuleSpec],
        
    ):
        super().__init__(wan_clip_config)
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            config = wan_clip_config,
            wan_clip_attn_spec = wan_clip_attn_spec,
            pretrained = False,
            return_transforms = True,
            return_tokenizer = False,
        )
        
    def forward(self, videos, attention_mask):

        size = (self.model.image_size,) * 2
        videos = torch.cat([
            F.interpolate(
                u.transpose(0, 1),
                size=size,
                mode='bicubic',
                align_corners=False) for u in videos
        ])

        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward

        with torch.cuda.amp.autocast(dtype=self.dtype):
            out = self.model.visual(videos, attention_mask, use_31_block=True)
            return out
    