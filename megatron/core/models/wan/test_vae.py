from wan_vae import WanVae
import torch
import os
from torch.optim import Adam
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core import parallel_state

def initialize_distributed(tensor_model_parallel_size = 1, pipeline_model_parallel_size = 1):
    # Torch setup for distributed training
    rank = int(os.environ['LOCAL_RANK'])
    world_size = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(world_size=world_size, rank=rank)

    # Megatron core distributed training initialization
    parallel_state.initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size)
    print("finish initialize_model")

def test_vae():
    weight_dtype = torch.float16
    initialize_distributed()
    model_parallel_cuda_manual_seed(42)
    config = TransformerConfig(num_layers=1, hidden_size=1, num_attention_heads=1)
    pretrained_model_path = "/nas/njw1123/Wan2.1_VAE.pth"  # Update with actual path if needed
    vae_model = WanVae(config = config, pretrained_model_path=pretrained_model_path).to(weight_dtype)
    vae_model.requires_grad_(False)
    device = torch.device("cuda")
    vae_model.to(device)
    input = torch.randn(1, 3, 21, 720, 1248).to(weight_dtype).to(device)
    output = vae_model.encode(input)
    print(output)
    print(output[0].sample().shape)
    

if __name__ == "__main__":
    # This is a placeholder for running the WANVAE model directly.
    # You can add code here to test or run the model if needed.
    print("WANVAE module loaded. You can now use WANVAE class.")
    test_vae()
    
    
    