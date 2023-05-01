import torch
import os
import numpy as np
import torchmetrics
from tqdm import tqdm
from utils import apply_heuristics, get_unique_objects

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)

hidden_data_path = "/mnt/d/Downloads/hidden_masks/val"
tensor_path = "./masks/stacked_pred_no_h.pt"
stacked_pred = torch.load(tensor_path)

print("Predictions: ", stacked_pred.shape)

# Read all masks from the hidden data path
video_masks = []
for video in tqdm(os.listdir(hidden_data_path)):
    mask_path = os.path.join(hidden_data_path, video, "mask.npy")
    mask = np.load(mask_path)
    video_masks.append(mask)

video_masks = np.array(video_masks)
input_img_masks = video_masks[:, :11]
gt_mask = video_masks[:, -1]

print("Hidden segmentation: ", video_masks.shape)
unique_original_objects = get_unique_objects(input_img_masks)

fixed_stacked_pred = apply_heuristics(stacked_pred,
                                      unique_original_objects, 'connected_components')

print("Fixed Predictions: ", fixed_stacked_pred.shape)
jaccard_val = jaccard(torch.Tensor(fixed_stacked_pred),
                      torch.Tensor(gt_mask))
print("mIoU: ", jaccard_val)
