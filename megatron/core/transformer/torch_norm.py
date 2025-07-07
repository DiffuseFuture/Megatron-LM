# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch

from torch import nn

from megatron.core.jit import jit_fuser
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.core import parallel_state
from megatron.core.utils import divide, get_tensor_model_parallel_group_if_none, is_torch_min_version
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from typing import Optional


class WrappedTorchNorm:
    """
    A conditional wrapper to initialize an instance of PyTorch's
    `LayerNorm` or `RMSNorm` based on input
    """

    def __new__(
        cls,
        config: TransformerConfig,
        hidden_size: int,
        eps: float = 1e-5,
        # TODO: unused arguments.
        # See https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/223
        persist_layer_norm: bool = False,
        zero_centered_gamma: bool = False,
        normalization: str = "LayerNorm",
    ):
        assert (
            not config.layernorm_zero_centered_gamma
        ), f"zero_centered_gamma not supported by torch LayerNorm"

        assert not config.persist_layer_norm, f"persist_layer_norm not supported by torch LayerNorm"

        assert not config.sequence_parallel, f"sequence parallel not supported by torch LayerNorm"

        assert (
            not config.memory_efficient_layer_norm
        ), f"memory_efficient_layer_norm not supported by torch LayerNorm"

        if config.normalization == "LayerNorm":
            norm_cls = torch.nn.LayerNorm
        elif config.normalization == "RMSNorm":
            assert is_torch_min_version(
                "2.4.0a0"
            ), 'Torch RMSNorm requires PyTorch version >= 2.4.0'

            norm_cls = torch.nn.RMSNorm
        elif config.normalization == "L2Norm":
            norm_cls = torch.nn.L2Norm
        else:
            raise Exception("Only LayerNorm, RMSNorm and L2Norm are currently supported")

        return norm_cls(normalized_shape=hidden_size, eps=eps)


class L2Norm(torch.nn.Module):
    """
    Applies L2 normalization to the input tensor along the last dimension.

    This module normalizes the input tensor such that the mean of the squared values
    along the last dimension is 1 (within a small epsilon for numerical stability).

    Args:
        hidden_size (int): Expected input shape for normalization (not used internally).
        eps (float, optional): A small value added to the denominator for numerical stability.
            Default: 1e-6.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    @jit_fuser
    def _norm(self, x):
        """
        Performs the actual L2 normalization.

        Args:
            x (torch.Tensor): The input tensor to normalize.

        Returns:
            torch.Tensor: The L2-normalized tensor.
        """
        x_float = x.float()
        return (x_float * torch.rsqrt(x_float.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)

    def forward(self, x):
        """
        Forward pass of the L2Norm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: L2-normalized tensor with the same dtype as input.
        """
        return self._norm(x)



# class WanRMSNorm(torch.nn.Module):

#     def __init__(self, config, hidden_size):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.eps = config.layernorm_epsilon
#         self.weight = nn.Parameter(torch.ones(hidden_size))

#     def forward(self, x):
#         r"""
#         Args:
#             x(Tensor): Shape [B, L, C]
#         """
#         return self._norm(x.float()).type_as(x) * self.weight

#     def _norm(self, x):
#         return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(torch.nn.LayerNorm):

    def __init__(self, config, hidden_size):
        super().__init__(hidden_size, elementwise_affine=False, eps=config.layernorm_epsilon)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)




class WanRMSNorm(torch.nn.Module):

    def __init__(self,
                 config,
                 hidden_size,
                 tp_group: Optional[torch.distributed.ProcessGroup] = None,):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = config.layernorm_epsilon
        self.tp_group = tp_group
        self.weight = nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, S, H, D] or [S, B, H, D]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        hidden_size = x.size(-1)
        assert hidden_size == self.hidden_size
        global_mean = None

        if self.tp_group is not None:
            local_mean = x.pow(2).mean(dim = -1,  keepdim=True) / self.tp_group.size()
            global_mean = reduce_from_tensor_model_parallel_region(local_mean, group=self.tp_group)
        else:
            global_mean = x.pow(2).mean(dim=-1, keepdim=True) 
        
        return x * torch.rsqrt(global_mean + self.eps)
