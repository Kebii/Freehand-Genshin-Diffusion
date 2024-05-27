import json
import random
import os

import torch
import torchvision.transforms as transforms
from decord import VideoReader
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class GenshinVdImgDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        srcdata_path="./Genshin_PV/raw",
        tgtdata_path="./Genshin_PV/draw"
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.srcdata_path = srcdata_path
        self.tgtdata_path = tgtdata_path
        self.tgt_videofile_lst = os.listdir(tgtdata_path)
        # self.tgt_videoname_lst = [vf[:-4] for vf in self.tgt_videofile_lst]        # remove the .mp4 suffix
        # self.src_videofile_lst = [vf+"_ori.mp4" for vf in self.tgt_videoname_lst]

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        tgt_videofile = self.tgt_videofile_lst[index]
        src_videofile = tgt_videofile[:-4] + "_ori.mp4"

        tgt_video_reader = VideoReader(os.path.join(self.tgtdata_path, tgt_videofile))
        src_video_reader = VideoReader(os.path.join(self.srcdata_path, src_videofile))

        assert len(tgt_video_reader) == len(
            src_video_reader
        ), f"{len(tgt_video_reader) = } != {len(src_video_reader) = } in {tgt_videofile}"

        video_length = len(tgt_video_reader)

        img_idx = random.randint(0, video_length - 1)
        tgt_img = tgt_video_reader[img_idx]
        tgt_img_pil = Image.fromarray(tgt_img.asnumpy())
        src_img = src_video_reader[img_idx]
        src_img_pil = Image.fromarray(src_img.asnumpy())

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        src_img = self.augmentation(src_img_pil, self.transform, state)
        
        sample = dict(
            tgt_videofile=tgt_videofile,
            tgt_img=tgt_img,
            src_img=src_img,
        )

        return sample

class GenshinImgDataset(Dataset):
    def __init__(
        self,
        img_size,
        img_scale=(1.0, 1.0),
        img_ratio=(0.9, 1.0),
        drop_ratio=0.1,
        srcdata_path="./Genshin_PV/raw_img",
        tgtdata_path="./Genshin_PV/draw_img"
    ):
        super().__init__()

        self.img_size = img_size
        self.img_scale = img_scale
        self.img_ratio = img_ratio

        self.srcdata_path = srcdata_path
        self.tgtdata_path = tgtdata_path
        self.img_file_lst = os.listdir(srcdata_path)

        self.clip_image_processor = CLIPImageProcessor()

        self.transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.img_size,
                    scale=self.img_scale,
                    ratio=self.img_ratio,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.drop_ratio = drop_ratio

    def augmentation(self, image, transform, state=None):
        if state is not None:
            torch.set_rng_state(state)
        return transform(image)

    def __getitem__(self, index):
        file_name = self.img_file_lst[index]
        tgt_img_pil = Image.open(os.path.join(self.tgtdata_path, file_name))
        src_img_pil = Image.open(os.path.join(self.srcdata_path, file_name))

        state = torch.get_rng_state()
        tgt_img = self.augmentation(tgt_img_pil, self.transform, state)
        src_img = self.augmentation(src_img_pil, self.transform, state)
        clip_image = self.clip_image_processor(
            images=src_img_pil, return_tensors="pt"
        ).pixel_values[0]
        
        sample = dict(
            file_name=file_name,
            tgt_img=tgt_img,
            src_img=src_img,
            clip_img=clip_image
        )

        return sample

    def __len__(self):
        return len(self.img_file_lst)
