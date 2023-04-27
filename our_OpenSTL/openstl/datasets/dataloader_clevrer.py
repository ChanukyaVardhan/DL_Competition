import gzip
import numpy as np
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset

from openstl.datasets.utils import create_loader


class Clevrer(Dataset):

    def __init__(self, params, transform, split = 'train', use_unlabeled = False):
        self.path           = params['path']
        self.num_frames     = params['num_frames']
        self.num_input_frames = params.get('num_input_frames', 11)
        self.to_predict     = params.get('to_predict', 11)
        self.num_samples    = params.get('num_samples', 0)
        self.img_height     = params.get('height', 160)
        self.img_width      = params.get('height', 240)
        self.img_channels   = params.get('channels', 3)
        self.transform      = transform

        # FIX : Possibly need to normalize the data.
        self.mean = 0
        self.std = 1

        if (split != 'train' and split != 'val'):
            raise ValueError("Split must be either 'train' or 'val'")
        
        self.data_path = os.path.join(self.path, split)

        self.video_paths    = []
        if use_unlabeled:
            up = os.path.join(self.path, "unlabeled")
            self.video_paths = self.video_paths + [os.path.join(up, v) for v in os.listdir(up) if os.path.isdir(os.path.join(up, v))]
        self.video_paths    = self.video_paths + [os.path.join(self.data_path, v) for v in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, v))]

        self.video_paths.sort()

        # self.video_paths = self.video_paths[:24]

        print("Videos before : ", len(self.video_paths))
        # print("***********TRAINING FOR A SINGLE VIDEO LOCALLY!**********")
        # self.video_paths = self.video_paths[0:300]
        # print("Videos after : ", len(self.video_paths))

    def __len__(self):
        return len(self.video_paths) if self.num_samples == 0 else min(self.num_samples, len(self.video_paths))

    def _load_image(self, image_path):
        image           = Image.open(image_path)
        image           = self.transform(image) if self.transform is not None else image

        return image

    def __getitem__(self, index):
        video_path      = self.video_paths[index]
    
        input_frames = [i for i in range(self.num_input_frames)]

        output_frames = [i for i in range(self.num_input_frames, self.num_input_frames + self.to_predict)]

        input_images    = []
        for index in input_frames:
            image       = self._load_image(os.path.join(video_path, f"image_{index}.png"))
            input_images.append(image)
        input_images    = torch.stack(input_images, dim = 0)

        output_images    = []
        for index in output_frames:
            image       = self._load_image(os.path.join(video_path, f"image_{index}.png"))
            output_images.append(image)
        output_images    = torch.stack(output_images, dim = 0)
        
        # order: num_frames(0) x img_channel(1) x img_height(2) x img_width(3) : 22 x 3 x 160 x 240
        return input_images, output_images
    

def load_data(batch_size, val_batch_size,
                params, transform,
               data_root=None, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, distributed=False):

    train_set = Clevrer(params=params, transform=transform, split = 'train', use_unlabeled = True)
    val_set = Clevrer(params=params, transform=transform, split = 'val', use_unlabeled = False)

    # FIX : Looks correct, but look carefully all the parameters of create_loader later.
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers, distributed=distributed)
    dataloader_vali = create_loader(val_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=False,
                                    num_workers=num_workers, distributed=distributed)
    # dataloader_test = create_loader(test_set,
    #                                 batch_size=val_batch_size,
    #                                 shuffle=False, is_training=False,
    #                                 pin_memory=True, drop_last=True,
    #                                 num_workers=num_workers, distributed=distributed)

    # FIX : Change this later maybe.
    return dataloader_train, dataloader_vali, None
