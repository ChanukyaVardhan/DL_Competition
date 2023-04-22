# https://github.com/wenwei202/pytorch-examples/blob/autogrow/cifar10/get_mean_std.py

from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import os
import torch
import torchvision.transforms as transforms

class TestDataset(Dataset):
    
    def __init__(self, data_dir = './data/'):
        self.data_dir = data_dir
        
        self.path = self.data_dir + "unlabeled"
        self.video_paths = [os.path.join(self.path, v) for v in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, v))]
        self.path = self.data_dir + "train"
        self.video_paths = self.video_paths + [os.path.join(self.path, v) for v in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, v))]
        self.video_paths.sort()

        # self.video_paths = self.video_paths[6900:]
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    # 13268, 13534, 13688, 14125
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, index):
        video_path = self.video_paths[index]

        images = []
        for idx in np.arange(22):
            img_path = os.path.join(video_path, f"image_{idx}.png")
            img = self.transform(Image.open(img_path))
            images.append(img)
        image_tensor = torch.stack(images, dim = 0)
        
        return image_tensor, video_path

trainset = TestDataset()

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, num_workers=2)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
h, w = 0, 0
for batch_idx, (inputs, vpath) in enumerate(trainloader):
    print(vpath)
    if (batch_idx+1) % 100 == 0:
        print(f"Completed mean for {(batch_idx+1)} videos!")
    
    inputs = inputs.to(device)
    inputs = inputs.squeeze()
    if batch_idx == 0:
        h, w = inputs.size(2), inputs.size(3)
        print(inputs.min(), inputs.max())
        chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
mean = chsum/(len(trainset) * 22)/h/w
print('mean: %s' % mean.view(-1))

chsum = None
for batch_idx, (inputs, vpath) in enumerate(trainloader):
    
    if (batch_idx+1) % 100 == 0:
        print(f"Completed std for {(batch_idx+1)} videos!")
        
    inputs = inputs.to(device)
    inputs = inputs.squeeze()
    if batch_idx == 0:
        chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
    else:
        chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
std = torch.sqrt(chsum/(len(trainset) * 22 * h * w - 1))
print('std: %s' % std.view(-1))

print('Done!')
