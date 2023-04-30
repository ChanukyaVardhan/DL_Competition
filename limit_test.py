import os
import torch
import numpy as np
import torchmetrics
from utils import class_labels, get_class_name

data_dir = "/vast/snm6477/DL_Finals/Dataset_Student/val"
masks = []
for video_id in os.listdir(data_dir):
    mask = np.load(os.path.join(data_dir, video_id, "mask.npy"))
    masks.append(mask)

masks = np.array(masks)
print(masks.shape)
eleventh_frame = masks[:, 10]
gt_masks = masks[:, -1]

# For each of the frames in the eleventh frame array and gt masks, get the unique objects
# Now remove all new objects from gt_masks that are not present in eleventh frame
limit_gt = []
for el_frame, gt_frame in zip(eleventh_frame, gt_masks):
    eleven_unique = np.unique(el_frame)
    gt_unique = np.unique(gt_frame)
    to_remove = np.setdiff1d(gt_unique, eleven_unique)
    removed_gt = gt_frame.copy()
    removed_gt[np.isin(removed_gt, to_remove)] = 0
    limit_gt.append(removed_gt)

limit_gt = np.array(limit_gt)

jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
jaccard_val = jaccard(torch.Tensor(limit_gt), torch.Tensor(gt_masks))
print(jaccard_val)
