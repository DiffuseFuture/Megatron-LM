
from bucket_sampler import (ASPECT_RATIO_512,
                            ASPECT_RATIO_RANDOM_CROP_512,
                            ASPECT_RATIO_RANDOM_CROP_PROB,
                            AspectRatioBatchImageVideoSampler,
                            RandomSampler, get_closest_ratio)

from dataset_image_video import (ImageVideoDataset,
                                                 ImageVideoSampler,
                                                 get_random_mask)

import torch
from argparse import Namespace
import numpy as np
from torchvision import transforms




def test():
    args = Namespace()
    args.train_data_meta="/root/add_dit/test_data/test.json"
    args.train_data_dir="/root/add_dit/test_data"
    args.video_sample_size=960
    args.token_sample_size=512
    args.video_sample_stride=2
    args.video_sample_n_frames=81
    args.video_repeat=1
    args.image_sample_size=1024
    args.enable_bucket=True
    args.seed = 22
    args.random_hw_adapt = True
    args.training_with_video_token_length = True
    args.train_mode="inpaint" 
    args.random_ratio_crop = False
    args.enable_text_encoder_in_dataloader = False
    
    train_dataset = ImageVideoDataset(
        args.train_data_meta, args.train_data_dir,
        video_sample_size=args.video_sample_size, video_sample_stride=args.video_sample_stride, video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        image_sample_size=args.image_sample_size,
        enable_bucket=args.enable_bucket, enable_inpaint=True,
    )
    
    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    # dp_size = parallel_state.get_data_parallel_world_size()
    # mini_batch_size = args.micro_batch_size * dp_size
    mini_batch_size = 1


    aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
    batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    batch_sampler = AspectRatioBatchImageVideoSampler(
        sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=train_dataset.dataset, 
        batch_size=mini_batch_size, train_folder = args.train_data_dir, drop_last=True,
        aspect_ratios=aspect_ratio_sample_size,
    )

    sample_n_frames_bucket_interval = 4


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

            print("length_to_frame_num", length_to_frame_num)

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
        # worker_init_fn=worker_init_fn(args.seed)
    )

    return train_dataloader


if __name__ == "__main__":
    data_loader = test()
    for batch in data_loader:
        print("====== New Batch ======")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
            elif isinstance(value, list):
                print(f"{key}: list of length {len(value)}")
            else:
                print(f"{key}: {type(value)}")
        # break  # Remove this break if you want to see more than one batch