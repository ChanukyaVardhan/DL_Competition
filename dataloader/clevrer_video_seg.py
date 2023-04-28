from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import masks_to_boxes

import numpy as np
import os
import torch
import torchvision.transforms as transforms


# Dataset to be used for training the final segmentation model.
class CLEVRERVideoSeg(Dataset):

    def __init__(self, data_dir="./data", split="train", transform=transforms.ToTensor()):
        self.data_dir = data_dir
        self.split = split 		# train/unlabeled/val/test
        self.path = os.path.join(self.data_dir, self.split)
        self.start_frame = 0
        self.num_frames = 11
        self.transform = transform

        self.video_ids = sorted([v for v in os.listdir(
            self.path) if os.path.isdir(os.path.join(self.path, v))])
        # FIX - UNCOMMENT THIS TO RUN LOCALLY
        self.video_ids = self.video_ids[:5]

    def __len__(self):
        return len(self.video_ids)

    def _load_image(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image) if self.transform is not None else image

        return image

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        video_path = os.path.join(self.path, video_id)
        mask_path = os.path.join(video_path, "mask.npy")

        # Get consecutive sample_frames number of frames
        input_images = []
        for index in range(self.num_frames):
            image = self._load_image(os.path.join(
                video_path, f"image_{index}.png"))
            input_images.append(image)
        input_images = torch.stack(input_images, dim=0)

        return input_images, mask_path
