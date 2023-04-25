from collections import OrderedDict
from PIL import Image
import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from convttlstm.utils.convlstmnet import ConvLSTMNet
from segmentation import SegNeXT
import torchmetrics

class TEST_Dataset(Dataset):
    def __init__(self, data_dir = "./data", num_samples = 0, transform = None, split = 'test'):
        self.data_dir       = data_dir
        self.num_samples    = num_samples
        self.transform      = transform
        self.split          = split
        self.path           = os.path.join(self.data_dir, self.split)

        # CHECK - THE VIDEO PATHS SHOULD BE SORTED I GUESS?
        self.video_paths    = [os.path.join(self.path, v) for v in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, v))]
        self.video_paths.sort()

    def __len__(self):
        return len(self.video_paths) if self.num_samples == 0 else min(self.num_samples, len(self.video_paths))

    def _load_image(self, image_path):
        image           = Image.open(image_path)
        image           = self.transform(image) if self.transform is not None else image

        return image

    def _load_images(self, video_path, frames):
        images    = []
        for index in frames:
            image       = self._load_image(os.path.join(video_path, f"image_{index}.png"))
            images.append(image)
        images    = torch.stack(images, dim = 0)

        return images

    def __getitem__(self, index):
        video_path      = self.video_paths[index]

        if self.split == "test": # LOAD THE 11 FRAMES, AND RETURN 0's FOR OTHERS
            frames = range(0, 11)
            target_images = torch.zeros(11, 3, 160, 240)
            gt_mask = torch.zeros(11, 160, 240)
        elif self.split == "train" or self.split == "val": # LOAD ALL FRAMES AND THE MASK
            frames = range(0, 11)
            target_frames = range(11, 22)
            gt_mask = torch.tensor(np.load(os.path.join(video_path, "mask.npy")))
        else: # UNLABELED -> RETURN 22 FRAMES
            frames = range(0, 11)
            target_frames = range(11, 22)
            gt_mask = torch.zeros(11, 160, 240)

        input_images = self._load_images(video_path, frames)
        if self.split != "test":
            target_images = self._load_images(video_path, target_frames)

        _, video_name = os.path.split(video_path)

        # input_images -> (11, 3, 160, 240); target_images -> (11, 3, 160, 240); gt_mask -> (11, 160, 240)
        return video_name, input_images, target_images, gt_mask

class FINAL_Model(nn.Module):
    def __init__(self, video_predictor = "convttlstm", video_predictor_path = "./checkpoints",
                       segmentation = "segnext", segmentation_path = "./checkpoints",
                       split = "test"):
        super(FINAL_Model, self).__init__()

        self.video_predictor = video_predictor
        self.video_predictor_path = video_predictor_path
        self.segmentation = segmentation
        self.segmentation_path = segmentation_path
        self.split = split

        if self.video_predictor == "convttlstm":
            self.m1 = ConvLSTMNet(
                input_channels = 3, 
                output_sigmoid = False,
                # model architecture
                layers_per_block = (3, 3, 3, 3), 
                hidden_channels  = (32, 48, 48, 32), 
                skip_stride = 2,
                # convolutional tensor-train layers
                cell = "convttlstm",
                cell_params = {
                    "order": 3,
                    "steps": 3,
                    "ranks": 8},
                # convolutional parameters
                kernel_size = 5)
            # SINCE CONVTTLSTM MODEL WAS SAVED USING DDP, NEED TO REMVOE MODULE FORM KEYS.
            # NEXT TIME SAVE MODEL.MODULE.SAVE_DICT() INSTEAD OF MODEL.SAVE_DICT() WHEN USING DDP.
            state_dict = torch.load(self.video_predictor_path, map_location='cpu')["model"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            self.m1.load_state_dict(new_state_dict)
            # self.m1 = DDP(self.m1, device_ids = [0])
            # self.m1.load_state_dict(torch.load(self.video_predictor_path, map_location = "cuda")["model"])
            self.m1.eval()
            print("Loaded convttlstm model!")
        else:
            raise Exception("FIX THIS!")

        if self.segmentation == "segnext":
            self.seg = SegNeXT(49, weights=None)
            self.seg.load_state_dict(torch.load(self.segmentation_path)["model"])
            self.seg.eval()
            print("Loaded segmentation model!")
        else:
            raise Exception("FIX THIS!")

    def forward(self, input_images, target_images):
        # input_images -> B, 11, 3, 160 ,240
        if self.video_predictor == "convttlstm":
            pred_images = self.m1(input_images,
                                  input_frames = 11, future_frames = 11, output_frames = 11, teacher_forcing = False)
            pred_image = pred_images[:, -1]
        else:
            raise Exception("FIX THIS!")

        target_image = target_images[:, -1]

        if self.segmentation == "segnext":
            pred_mask = self.seg(pred_image)
            pred_mask = torch.argmax(pred_mask, dim=1)
            if self.split != "test" and self.split != "unlabeled": # WE CAN COMPUTE THE SEGMENTATION OUTPUT ONLY ON TRAIN/VAL VIDEOS
                target_mask = self.seg(target_image)
                target_mask = torch.argmax(target_mask, dim=1)
            else:
                target_mask = None
        else:
            raise Exception("FIX THIS!")

        return pred_mask, target_mask

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
                         0.0571, 0.0567, 0.0614])
])
split = "val" # WE CAN CHANGE TO TRAIN/VAL/UNLABELED AS WELL
num_samples = 100 # 0 MEANS USE THE WHOLE DATASET
data_dir = "./data"
video_predictor = "convttlstm"
video_predictor_path = "./checkpoints/convttlstm_best.pt"
segmentation = "segnext"
segmentation_path = "./checkpoints/segmentation_default_pretrain_model.pt"

dataset = TEST_Dataset(data_dir = data_dir, num_samples = num_samples, transform = transform, split = split)

batch_size = 4
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, drop_last = False, num_workers = 4, shuffle = False)

print(f"Number of total samples = {len(dataset)}")

model = FINAL_Model(video_predictor = video_predictor, video_predictor_path = video_predictor_path,
                    segmentation = segmentation, segmentation_path = segmentation_path,
                    split = split).cuda()
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters - {total_params}")

stacked_pred = [] # stacked predicted segmentation of predicted 22nd frame
stacked_target = [] # stacked predicted segmentation of actual 22nd frame (only if not test)
stacked_gt = [] # stacked actual segmentation (only if train/val)

with torch.no_grad():
    model.eval()
    
    for it, (_, input_images, target_images, gt_mask) in enumerate(dataloader):
        input_images = input_images.cuda()
        target_images = target_images.cuda()
        gt_mask = gt_mask[:, -1].cuda()

        pred_mask, target_mask = model(input_images, target_images)

        stacked_pred.append(pred_mask.cpu())
        if split !="test" and split != "unlabeled":
            stacked_target.append(target_mask.cpu())
            stacked_gt.append(gt_mask.cpu())

    stacked_pred = torch.cat(stacked_pred,0)
    print(f"Stacked Pred shape - {stacked_pred.shape}")
    if split !="test" and split != "unlabeled":
        stacked_target = torch.cat(stacked_target,0)
        print(f"Stacked Orig shape - {stacked_target.shape}")
        stacked_gt = torch.cat(stacked_gt,0)
        print(f"Stacked GT shape - {stacked_gt.shape}")

    if split != 'test':
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        jaccard_val = jaccard(stacked_pred, stacked_gt)
        print("Jaccard of predicted with gt: ", jaccard_val)
        jaccard_val = jaccard(stacked_target, stacked_gt)
        print("Jaccard of original with gt: ", jaccard_val)
