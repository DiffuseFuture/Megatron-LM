import torch
import torch.nn.functional as F
from megatron.training import get_args
from discrete_sampler import DiscreteSampling

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def get_timesteps_and_sigmas(noise_scheduler, bs, n_dim=4, dtype=torch.float32):
    args = get_args()
    args.train_sampling_steps = 1000
    idx_sampling = DiscreteSampling(args.train_sampling_steps, uniform_sampling=True)
    indices = idx_sampling(bs, generator=None).cuda()
    indices = indices.long().cpu()
    timesteps = noise_scheduler.timesteps[indices].cuda()
    sigmas = noise_scheduler.sigmas.to(dtype=dtype).cuda()
    schedule_timesteps = noise_scheduler.timesteps.cuda()
    timesteps = timesteps.cuda()
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return timesteps, sigma
