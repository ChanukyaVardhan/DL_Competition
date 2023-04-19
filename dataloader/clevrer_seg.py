import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CLEVRERSeg(Dataset):
    
    def _init_(self, data_dir = './data/', split = 'train'):
        self.data_dir = data_dir
        self.split    = split
        
        self.path = os.path.join(self.data_dir, self.split)
        self.video_paths = [os.path.join(self.path, v) for v in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, v))]
        self.video_paths.sort()
        
        self.image_paths = [os.path.join(vpath, f"image_{i}.png") for i in range(22) for vpath in self.video_paths]

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    def _len_(self):
        return len(self.image_paths)
    
    def _getitem_(self, index):
        image_path = self.image_paths[index]        
        video_path, image_name = os.path.split(image_path)
        
        image_idx = int(image_name.split("_")[1].split(".")[0])
        
        image = Image.open(image_path)
        mask = torch.FloatTensor(np.load(os.path.join(video_path, "mask.npy"))[image_idx])

        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return image, mask