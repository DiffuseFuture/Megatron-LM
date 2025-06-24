from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.autoencoders.vae import (DecoderOutput,
                                               DiagonalGaussianDistribution)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange
from megatron.core.transformer.module import MegatronModule
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler



class WANNoiseScheduler(MegatronModule):
    def __init__(self, config, additional_kwargs={}):
        super().__init__(config)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler(**additional_kwargs)

