import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import List

import av
import numpy as np
import torch
import torchvision
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline
from einops import repeat
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection

from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_genshinimg import GenshinImagePipeline
from src.utils.util import get_fps, read_frames, save_videos_grid

from decord import VideoReader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("-W", type=int, default=480)
    parser.add_argument("-H", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fps", type=int)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKL.from_pretrained(
        config.pretrained_vae_path,
    ).to("cuda", dtype=weight_dtype)

    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(dtype=weight_dtype, device="cuda")

    inference_config_path = config.inference_config
    infer_config = OmegaConf.load(inference_config_path)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        None,
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device="cuda")

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device="cuda")

    sched_kwargs = OmegaConf.to_container(infer_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"),
        strict=False,
    )
    reference_unet.load_state_dict(
        torch.load(config.reference_unet_path, map_location="cpu"),
    )

    pipe = GenshinImagePipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        scheduler=scheduler,
    )
    pipe = pipe.to("cuda", dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    time_str = datetime.now().strftime("%H%M")
    save_dir_name = f"{time_str}--seed_{args.seed}-{args.W}x{args.H}"

    save_dir = Path(f"output/{date_str}/{save_dir_name}")
    save_dir.mkdir(exist_ok=True, parents=True)

    video_reader = VideoReader(config.input_video_path)
    ref_name = config.input_video_path.split("/")[-1].replace(".mp4", "")
    num_frames = len(video_reader)
    print("Total frames: {}".format(num_frames) )

    gen_images = []
    ref_images = []
    img_transform = transforms.Compose(
                [transforms.Resize((height, width)), transforms.ToTensor()]
            )
    # for ref_image in video_reader:
    for ref_image_i in range(num_frames):
        ref_image = video_reader[ref_image_i]
        print("Processing {}/{} frame.".format(ref_image_i+1, num_frames))
        ref_image_pil = Image.fromarray(ref_image.asnumpy())

        image = pipe(
            ref_image_pil,
            width,
            height,
            20,
            3.5,
            generator=generator,
        ).images                                               # b c t h w
        gen_images.append(image)
        ref_image_tensor = img_transform(ref_image_pil)  # (c, h, w)
        ref_image_tensor = ref_image_tensor.unsqueeze(1).unsqueeze(0)
        ref_images.append(ref_image_tensor)

    
    video = torch.cat([torch.cat(ref_images, dim=2), torch.cat(gen_images, dim=2)], dim=0)
    save_videos_grid(
        video,
        f"{save_dir}/{ref_name}_{args.H}x{args.W}_{int(args.cfg)}_{time_str}.mp4",
        n_rows=2,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
