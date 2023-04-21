import os
import numpy as np
from PIL import Image
from torchvision.ops import masks_to_boxes
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MaskRCNNDataset(Dataset):

    def __init__(self, data_dir='../data/', split='train', transforms=transforms.ToTensor()):
        self.data_dir = data_dir
        self.split = split
        if (split != "train" and split != "val"):
            raise ValueError("split must be either 'train' or 'val'")

        self.path = os.path.join(self.data_dir, self.split)
        self.video_paths = [os.path.join(self.path, v) for v in os.listdir(
            self.path) if os.path.isdir(os.path.join(self.path, v))]
        self.video_paths.sort()

        # self.video_paths = self.video_paths[0:1]  # Uncomment to test on a single video

        self.image_paths = [os.path.join(vpath, f"image_{i}.png") for i in range(
            22) for vpath in self.video_paths]

        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def _get_bounding_boxes(self, mask):
        # Get bounding boxes for each object in the mask
        # Input: mask - (H, W) tensor of values in [0, 48]
        # Returns a tensor of bounding boxes of shape [N, x1, y1, x2, y2]
        # where (x1, y1) is the top left corner and (x2, y2) is the bottom right corner
        # of the bounding box and N is the number of objects present in the mask. N <= 48.
        # Also returns the masks in a one-hot encoding fashion.
        obj_ids = torch.unique(mask)

        # first id is the background, so remove it.
        obj_ids = obj_ids[1:].long()

        masks = mask == obj_ids[:, None, None]
        # print("img : ", mask.shape)

        # print("objss : ", obj_ids.shape)

        boxes = masks_to_boxes(masks).long()
        return masks, obj_ids, boxes

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        video_path, image_name = os.path.split(image_path)
        image_idx = int(image_name.split("_")[1].split(".")[0])

        image = Image.open(image_path)
        mask = torch.FloatTensor(
            np.load(os.path.join(video_path, "mask.npy"))[image_idx])

        masks, labels, bounding_boxes = self._get_bounding_boxes(mask)

        # labels are the values of the objects. 1 <= values <= 48.
        return self.transforms(image), masks, labels, bounding_boxes


def mrcnn_collate_fn(batch):
    transposed = list(zip(*batch))
    targets = [{"masks": transposed[1][i],
                "labels": transposed[2][i],
                "boxes": transposed[3][i]} for i in range(len(transposed[0]))]
    # print(transposed)
    return torch.stack(transposed[0]), targets
