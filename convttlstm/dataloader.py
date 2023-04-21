# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license

from __future__ import print_function
from __future__ import division
from PIL import Image
import os.path
import glob

import math
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

class OUR_Dataset(Dataset):

    def __init__(self, params, transform, use_unlabeled = False, predict_final = False, predict_alternate = False):
        self.path           = params['path']
        self.num_frames     = params['num_frames']
        self.num_samples    = params.get('num_samples', 0)
        self.img_height     = params.get('height', 160)
        self.img_width      = params.get('height', 240)
        self.img_channels   = params.get('channels', 3)
        self.transform      = transform
        self.predict_final  = predict_final
        self.predict_alternate = predict_alternate

        self.video_paths    = []
        if use_unlabeled:
            p, _ = os.path.split(self.path)
            up = os.path.join(p, "unlabeled")
            self.video_paths = self.video_paths + [os.path.join(up, v) for v in os.listdir(up) if os.path.isdir(os.path.join(up, v))]
        self.video_paths    = self.video_paths + [os.path.join(self.path, v) for v in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, v))]

        self.video_paths.sort()

    def __len__(self):
        return len(self.video_paths) if self.num_samples == 0 else min(self.num_samples, len(self.video_paths))

    def _load_image(self, image_path):
        image           = Image.open(image_path)
        image           = self.transform(image) if self.transform is not None else image

        return image

    def __getitem__(self, index):
        video_path      = self.video_paths[index]

        # FIX THIS FOR TEST SET
        if self.predict_final:
            frames = [i for i in range(0, 11)] + [21]
        elif self.predict_alternate:
            frames = [i for i in range(0, 11)] + [11, 13, 15, 17, 19, 21]
        else:
            frames = [i for i in range(self.num_frames)] # all 22 frames

        input_images    = []
        for index in frames:
            image       = self._load_image(os.path.join(video_path, f"image_{index}.png"))
            input_images.append(image)
        input_images    = torch.stack(input_images, dim = 0)

        # 4th order: num_frames(0) x img_height(1) x img_width(2) x img_channel(3) : 22 x 160 x 240 x 3
        input_images    = input_images.permute(0, 2, 3, 1)

        return input_images

class MNIST_Dataset(Dataset):

    def __init__(self, params):
        # parameters of the dataset 
        path = params['path']
        assert os.path.exists(path), "The file does not exist."

        self.num_frames  = params['num_frames']
        self.num_samples = params.get('num_samples', None)

        self.random_crop = params.get('random_crop', False) 

        self.img_height   = params.get('height',  64)
        self.img_width    = params.get('width',   64)
        self.img_channels = params.get('channels', 1)

        self.data = np.load(path)["data"]
        self.data_samples = self.data.shape[0]
        self.data_frames  = self.data.shape[1]

    def __getitem__(self, index):
        start = random.randint(0, self.data_frames - 
            self.num_frames) if self.random_crop else 0

        data  = np.float32(self.data[index, start : start + self.num_frames] / 255.0)
        return data 

    def __len__(self):
        return len(self.data_samples) if self.num_samples is None \
            else min(self.data_samples,  self.num_samples)


class KTH_Dataset(Dataset):

    def __init__(self, params):
        # parameters of the dataset
        path = params['path']
        assert os.path.exists(path), "The dataset folder does not exist."

        unique_mode = params.get('unique_mode', True)

        self.num_samples = params.get('num_samples', None)
        self.num_frames  = params['num_frames']

        self.img_height   = params.get('height', 120)
        self.img_width    = params.get('width',  120)
        self.img_channels = params.get('channels', 3)

        self.training     = params.get('training', False)
        
        # parse the files in the data folder
        if self.training:
            self.files = glob.glob(os.path.join(path, '*.npz*'))
        else:
            self.files = sorted(glob.glob(os.path.join(path, '*.npz*')))

        self.clips = []
        for i in range(len(self.files)):
            data = np.load(self.files[i])["data"]
            data_frames = data.shape[0] 

            self.clips += [(i, t) for t in range(data_frames - self.num_frames)] if not unique_mode \
                else [(i, t * self.num_frames) for t in range(data_frames // self.num_frames)]

        self.data_samples = len(self.clips)

    def __getitem__(self, index):
        (file_index, start_frame) = self.clips[index]

        # 4th order: num_frames(0) x img_height(1) x img_width(2) x img_channel(3)
        data = np.load(self.files[file_index])["data"]

        # place holder for data processing
        _, img_height, img_width, _ = data.shape
        if img_height == self.img_height and img_width == self.img_width:
            clip = data[start_frame : start_frame + self.num_frames]
        else: # resizing the input is needed
            clip = np.stack([resize(data[start_frame + t], (self.img_height, self.img_width)) 
                for t in range(self.num_frames)], axis = 0)

        data = np.float32(clip)
        if self.img_channels == 1:
            data = np.mean(data, axis = -1, keepdims = True)

        return data.astype(np.float32)

    def __len__(self):
        return self.data_samples if self.num_samples is None \
            else min(self.data_samples, self.num_samples)