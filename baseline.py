import os
import numpy as np
import torch
import torchmetrics

data_dir = "./Dataset_Student/train"
video_masks = []
for video in os.listdir(data_dir):
    video_dir = os.path.join(data_dir, video)
    mask_path = os.path.join(video_dir, "mask.npy")
    mask = np.load(mask_path)
    video_masks.append(mask)

video_masks = np.array(video_masks)

eleven_mask = video_masks[:, 10, :, :]
gt_mask = video_masks[:, 21, :, :]

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
jaccard = jaccard(torch.Tensor(eleven_mask), torch.Tensor(gt_mask))

print("mIoU: ", jaccard)
