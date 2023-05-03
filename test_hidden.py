import matplotlib.colors as mcolors
from collections import OrderedDict
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from our_OpenSTL.openstl.models import SimVP_Model, Decoder
import torchmetrics
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import class_labels, get_unique_objects, apply_heuristics

simvp_config = {
    "in_shape": [11, 3, 160, 240],
    "hid_S": 64,
    "hid_T": 512,
    "N_S": 4,
    "N_T": 8,
    "spatio_kernel_enc": 3,
    "spatio_kernel_dec": 3,
    "num_classes": 49,
}


class TEST_Dataset(Dataset):
    def __init__(self, data_dir="./data", num_samples=0, transform=None, split='test'):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.transform = transform
        self.split = split
        self.path = os.path.join(self.data_dir, self.split)

        # _______IMPORTANT CHECK - THE VIDEO PATHS SHOULD BE SORTED I GUESS?
        self.video_paths = [os.path.join(self.path, v) for v in os.listdir(
            self.path) if os.path.isdir(os.path.join(self.path, v))]
        self.video_paths.sort()

    def __len__(self):
        return len(self.video_paths) if self.num_samples == 0 else min(self.num_samples, len(self.video_paths))

    def _load_image(self, image_path):
        image = Image.open(image_path)
        image = self.transform(image) if self.transform is not None else image

        return image

    def _load_images(self, video_path, frames):
        images = []
        for index in frames:
            image = self._load_image(os.path.join(
                video_path, f"image_{index}.png"))
            images.append(image)
        images = torch.stack(images, dim=0)

        return images

    def __getitem__(self, index):
        video_path = self.video_paths[index]

        if self.split == "hidden":  # LOAD THE 11 FRAMES, AND RETURN 0's FOR OTHERS
            frames = range(0, 11)
            target_images = torch.zeros(11, 3, 160, 240)
            gt_mask = torch.tensor(
                np.load(os.path.join(video_path, "mask.npy")))
        elif self.split == "train" or self.split == "val":  # LOAD ALL FRAMES AND THE MASK
            frames = range(0, 11)
            target_frames = range(11, 22)
            gt_mask = torch.tensor(
                np.load(os.path.join(video_path, "mask.npy")))
        else:  # UNLABELED -> RETURN 22 FRAMES
            frames = range(0, 11)
            target_frames = range(11, 22)
            gt_mask = torch.zeros(11, 160, 240)

        input_images = self._load_images(video_path, frames)
        if self.split != "hidden":
            target_images = self._load_images(video_path, target_frames)

        _, video_name = os.path.split(video_path)

        # input_images -> (11, 3, 160, 240); target_images -> (11, 3, 160, 240); gt_mask -> (11, 160, 240)
        return video_name, input_images, target_images, gt_mask


split = "hidden"  # WE CAN CHANGE TO TRAIN/VAL/UNLABELED AS WELL
num_samples = 0  # 0 MEANS USE THE WHOLE DATASET

data_dir = "/scratch/pj2251/DL/DL_Competition/data/Dataset_Student"

video_predictor = "ft_simvp"
video_predictor_path = "./ft_checkpoints/ft_simvp_segmentation_model_36_0.42.pth"
segmentation = "deeplabv3"
segmentation_path = "./checkpoints/deeplab_v3_segmentation_model_50.pth"

transform = transforms.Compose([transforms.ToTensor()])


dataset = TEST_Dataset(
    data_dir=data_dir, num_samples=num_samples, transform=transform, split=split)

batch_size = 16
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, drop_last=False, num_workers=4, shuffle=False)

print(f"Number of total samples = {len(dataset)}")

if video_predictor == "ft_simvp":
    model = SimVP_Model(**simvp_config)
    T, C, H, W = simvp_config["in_shape"]
    model.dec = Decoder(simvp_config["hid_S"], C,
                        simvp_config["N_S"], simvp_config["spatio_kernel_dec"])
    model.dec.readout = nn.Conv2d(
        simvp_config["hid_S"], simvp_config["num_classes"], kernel_size=1)
    model.load_state_dict(torch.load(video_predictor_path))
    model = model.cuda()
else:
    raise Exception("Invalid video predictor")

model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters - {total_params}")

stacked_pred = []  # stacked predicted segmentation of predicted 22nd frame
# stacked predicted segmentation of actual 22nd frame (only if not test)
stacked_target = []
stacked_gt = []  # stacked actual segmentation (only if train/val)

unique_original_objects = []

save_images = False  # set to True to save images

colors = np.random.rand(49, 3)
cmap = mcolors.ListedColormap(colors)

with torch.no_grad():
    model.eval()

    for it, (_, input_images, target_images, gt_mask) in tqdm(enumerate(dataloader)):
        input_images = input_images.cuda()  # B, T, C, H, W
        target_images = target_images.cuda()  # Zero for hidden
        # Loaded from mask.npy, used for heuristics
        input_img_masks = gt_mask[:, :11].cpu()
        gt_mask = gt_mask[:, -1]

        pred_mask = model(input_images)
        pred_mask = torch.argmax(pred_mask, dim=2)
        pred_mask = pred_mask[:, -1, :, :]

        if save_images:
            for b in range(pred_mask.shape[0]):
                hstacked = np.hstack(
                    [gt_mask[b].cpu().numpy(), pred_mask[b].cpu().numpy()])
                plt.imsave(f"./images/{it}_{b}.png", hstacked, cmap=cmap)

        unique_original_objects = unique_original_objects + get_unique_objects(
            input_img_masks.cpu().numpy())

        stacked_pred.append(pred_mask.cpu())
        if split != "hidden" and split != "unlabeled":
            stacked_gt.append(gt_mask.cpu())

    print("Number of Unique original objects: ", len(unique_original_objects))
    # Sanity check ::
    print("Unique original objects[0]: ", unique_original_objects[0])

    stacked_pred = torch.cat(stacked_pred, 0)
    print(f"Stacked Pred shape - {stacked_pred.shape}")

    if split != "hidden" and split != "unlabeled":
        stacked_gt = torch.cat(stacked_gt, 0)
        print(f"Stacked GT shape - {stacked_gt.shape}")

    if split != 'hidden':
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        jaccard_val = jaccard(stacked_pred, stacked_gt)
        torch.save(stacked_pred, "stacked_pred_no_h.pt")
        fixed_stacked_pred = apply_heuristics(stacked_pred,
                                              unique_original_objects, 'connected_components')
        jaccard_val_h = jaccard(fixed_stacked_pred, stacked_gt)
        print("Jaccard of predicted with gt: ", jaccard_val)
        print("Jaccard of predicted with gt after heuristics: ", jaccard_val_h)
        torch.save(fixed_stacked_pred, "stacked_pred_hidden_heur.pt")
        if (video_predictor != "ft_simvp"):
            jaccard_val = jaccard(stacked_target, stacked_gt)
            print("Jaccard of original with gt: ", jaccard_val)
        jaccard_gt = jaccard(stacked_gt, stacked_gt)
        print("Jaccard of gt with gt: ", jaccard_gt)
    else:
        # Split is hidden ____________FINAL__SUBMISSION____________
        torch.save(stacked_pred, "stacked_pred_hidden.pt")
        print("Saved stacked_pred_hidden.pt of size : ", stacked_pred.shape)
        fixed_stacked_pred = apply_heuristics(stacked_pred,
                                              unique_original_objects,
                                              'connected_components')
        torch.save(fixed_stacked_pred, "stacked_pred_hidden_heur.pt")
        print("Saved stacked_pred_hidden_heur.pt of size : ",
              fixed_stacked_pred.shape)
