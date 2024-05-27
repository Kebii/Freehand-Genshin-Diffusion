import json
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor

import os


class GenshinVideoDataset(Dataset):
    def __init__(
        self,
        sample_rate,
        n_sample_frames,
        width,
        height,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        srcdata_path="./Genshin_PV/raw_video",
        tgtdata_path="./Genshin_PV/draw_video"
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_sample_frames = n_sample_frames
        self.width = width
        self.height = height
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.srcdata_path = srcdata_path
        self.tgtdata_path = tgtdata_path
        self.tgt_videofile_lst = os.listdir(tgtdata_path)


        self.clip_image_processor = CLIPImageProcessor()

        self.pixel_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (height, width),
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, images, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        if isinstance(images, List):
            transformed_images = [transform(img) for img in images]
            ret_tensor = torch.stack(transformed_images, dim=0)  # (f, c, h, w)
        else:
            ret_tensor = transform(images)  # (c, h, w)
        return ret_tensor

    def __getitem__(self, index):
        tgt_videofile = self.tgt_videofile_lst[index]
        src_videofile = tgt_videofile[:-4] + "_ori.mp4"

        tgt_video_reader = VideoReader(os.path.join(self.tgtdata_path, tgt_videofile))
        src_video_reader = VideoReader(os.path.join(self.srcdata_path, src_videofile))

        # commend to avoid the length mismatch of src and tgt videos
        # assert len(tgt_video_reader) == len(
        #     src_video_reader
        # ), f"{len(tgt_video_reader) = } != {len(src_video_reader) = } in {tgt_videofile}"

        video_length = min(len(tgt_video_reader), len(src_video_reader))

        clip_length = min(
            video_length, (self.n_sample_frames - 1) * self.sample_rate + 1
        )
        start_margin = 0        # 180 for full video
        end_margin = 0
        start_idx = random.randint(start_margin, video_length - clip_length - end_margin)
        batch_index = np.linspace(
            start_idx, start_idx + clip_length - 1, self.n_sample_frames, dtype=int
        ).tolist()

        # read frames and kps
        src_pil_image_list = []
        tgt_pil_image_list = []
        for index in batch_index:
            img = src_video_reader[index]
            src_pil_image_list.append(Image.fromarray(img.asnumpy()))
            img = tgt_video_reader[index]
            tgt_pil_image_list.append(Image.fromarray(img.asnumpy()))


        # transform
        state = torch.get_rng_state()
        src_values_vid = self.augmentation(
            src_pil_image_list, self.pixel_transform, state
        )
        tgt_values_vid = self.augmentation(
            tgt_pil_image_list, self.pixel_transform, state
        )
       
        clip_imgs = self.clip_image_processor(
            images=src_pil_image_list, return_tensors="pt"
        ).pixel_values

        sample = dict(
            file_name=tgt_videofile,
            tgt_frames=tgt_values_vid,
            src_frames=src_values_vid,
            clip_frames=clip_imgs
        )

        return sample

    def __len__(self):
        return len(self.tgt_videofile_lst)
