import itertools
import matplotlib.colors as mcolors
from collections import OrderedDict
from PIL import Image
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from convttlstm.utils.convlstmnet import ConvLSTMNet
from our_OpenSTL.openstl.models import SimVP_Model, Decoder
from segmentation import SegNeXT
import torchmetrics
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import class_labels, get_unique_objects, apply_heuristics, class_labels_list
from sklearn.metrics import confusion_matrix
import wandb


def plot_images(pred_mask, gt_mask, pred_image, image):

    return wandb.Image(image, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
        "ground truth": {"mask_data": gt_mask, "class_labels": class_labels}
    }), wandb.Image(pred_image)


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

save_images = True  # set to True to save images
if save_images:
    os.makedirs(f"./images/", exist_ok=True)


class TEST_Dataset(Dataset):
    def __init__(self, data_dir="./data", num_samples=0, transform=None, split='test'):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.transform = transform
        self.split = split
        self.path = os.path.join(self.data_dir, self.split)

        # CHECK - THE VIDEO PATHS SHOULD BE SORTED I GUESS?
        self.video_paths = [os.path.join(self.path, v) for v in os.listdir(
            self.path) if os.path.isdir(os.path.join(self.path, v))]
        self.video_paths.sort()

        # self.video_paths = self.video_paths[:1]

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


class FINAL_Model(nn.Module):
    def __init__(self, video_predictor="convttlstm", video_predictor_path="./checkpoints",
                 segmentation="segnext", segmentation_path="./checkpoints",
                 split="test"):
        super(FINAL_Model, self).__init__()

        self.video_predictor = video_predictor
        self.video_predictor_path = video_predictor_path
        self.segmentation = segmentation
        self.segmentation_path = segmentation_path
        self.split = split

        # self.normalize = torch.nn.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
        self.normalize = transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
            0.0571, 0.0567, 0.0614])

        if self.video_predictor == "convttlstm":
            self.m1 = ConvLSTMNet(
                input_channels=3,
                output_sigmoid=False,
                # model architecture
                layers_per_block=(3, 3, 3, 3),
                hidden_channels=(32, 48, 48, 32),
                skip_stride=2,
                # convolutional tensor-train layers
                cell="convttlstm",
                cell_params={
                    "order": 3,
                    "steps": 3,
                    "ranks": 8},
                # convolutional parameters
                kernel_size=5)
            # SINCE CONVTTLSTM MODEL WAS SAVED USING DDP, NEED TO REMVOE MODULE FORM KEYS.
            # NEXT TIME SAVE MODEL.MODULE.SAVE_DICT() INSTEAD OF MODEL.SAVE_DICT() WHEN USING DDP.
            state_dict = torch.load(
                self.video_predictor_path, map_location='cpu')["model"]
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
        elif self.video_predictor == "simvp":
            self.m1 = SimVP_Model(**simvp_config)
            # self.m1.load_state_dict(torch.load(self.video_predictor_path, map_location = "cpu")["state_dict"])
            self.m1.load_state_dict(torch.load(
                self.video_predictor_path, map_location="cpu"))
            self.m1.eval()
            print("Loaded SimVP model!")
        else:
            raise Exception("FIX THIS!")

        if self.segmentation == "segnext":
            self.seg = SegNeXT(49, weights=None)
            self.seg.load_state_dict(torch.load(
                self.segmentation_path, map_location='cpu')["model"])
            self.seg.eval()
            print(f"Loaded {self.segmentation} segmentation model!")
        elif self.segmentation == "deeplabv3":
            from torchvision.models.segmentation import deeplabv3_resnet50
            self.seg = deeplabv3_resnet50(
                num_classes=49, weights_backbone=None)
            self.seg.load_state_dict(torch.load(
                self.segmentation_path, map_location='cpu'))
            self.seg.eval()
            print(f"Loaded {self.segmentation} segmentation model!")
        else:
            raise Exception("FIX THIS!")

    def forward(self, input_images, target_images):
        # input_images -> B, 11, 3, 160 ,240
        if self.video_predictor == "convttlstm":
            pred_images = self.m1(input_images,
                                  input_frames=11, future_frames=11, output_frames=11, teacher_forcing=False)
            pred_image = pred_images[:, -1]
            target_image = target_images[:, -1]
        elif self.video_predictor == "simvp":
            pred_images = self.m1(input_images)
            pred_image = pred_images[:, -1]
            target_image = target_images[:, -1]
            pred_image_unnormalized = pred_image.clone()
            pred_image = torch.stack(
                [self.normalize(pred_image[i]) for i in range(pred_image.shape[0])])
            target_image = torch.stack(
                [self.normalize(target_image[i]) for i in range(target_image.shape[0])])
        else:
            raise Exception("FIX THIS!")

        if self.segmentation == "segnext":
            pred_mask = self.seg(pred_image)
            pred_mask = torch.argmax(pred_mask, dim=1)
            # WE CAN COMPUTE THE SEGMENTATION OUTPUT ONLY ON TRAIN/VAL VIDEOS
            if self.split != "test" and self.split != "unlabeled":
                target_mask = self.seg(target_image)
                target_mask = torch.argmax(target_mask, dim=1)
            else:
                target_mask = None
        elif self.segmentation == "deeplabv3":
            pred_mask = self.seg(pred_image)['out']
            pred_mask = torch.argmax(pred_mask, dim=1)
            if self.split != "test" and self.split != "unlabeled":
                target_mask = self.seg(target_image)['out']
                target_mask = torch.argmax(target_mask, dim=1)
            else:
                target_mask = None
        else:
            raise Exception("FIX THIS!")

        return pred_mask, target_mask, pred_image_unnormalized


def compute_confusion_matrix(true, pred, num_classes):
    # Flatten the tensor and convert to numpy for easier manipulation
    true = true.view(-1).cpu().numpy()
    pred = pred.view(-1).cpu().numpy()

    # Compute confusion matrix with sklearn
    cm = confusion_matrix(true, pred, labels=range(num_classes))

    return cm


def plot_confusion_matrix(cm, file_name):
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()

    threshold = 1000
    np.fill_diagonal(cm, 0)
    error_pairs = np.argwhere(cm > threshold)
    errors = confusion_mat[error_pairs[:, 0], error_pairs[:, 1]]
    sorted_indices = np.argsort(errors)[::-1]

    for idx in sorted_indices:
        pair = error_pairs[idx]
        if pair[0] != pair[1]:
            print(
                f"Truth: {class_labels_list[pair[0]]}, Predicted: {class_labels_list[pair[1]]}, Errors: {errors[idx]}")

    tick_marks = np.arange(len(class_labels_list))
    plt.xticks(tick_marks, class_labels_list, rotation=90)
    plt.yticks(tick_marks, class_labels_list)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'./images/{file_name}.png')


split = "val"  # WE CAN CHANGE TO TRAIN/VAL/UNLABELED AS WELL
num_samples = 0  # 0 MEANS USE THE WHOLE DATASET

data_dir = "/vast/snm6477/DL_Finals/Dataset_Student/"

# video_predictor = "convttlstm"
# video_predictor_path = "./checkpoints/convttlstm_best.pt"
video_predictor = "ft_simvp"
video_predictor_path = "./checkpoints/simvp_checkpoint_unnormalized.pth"
video_predictor_path = "./checkpoints/simvp_checkpoint.pth"
video_predictor_path = "./checkpoints/ft_simvp_segmentation_model_20.pth"
segmentation = "deeplabv3"
segmentation_path = "./checkpoints/segmentation_default_pretrain_model.pt"
segmentation_path = "./checkpoints/deeplab_v3_segmentation_model_50.pth"

if video_predictor == "convttlstm":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
                             0.0571, 0.0567, 0.0614])
    ])
elif "simvp" in video_predictor:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
#                              0.0571, 0.0567, 0.0614])
#     ])
else:
    raise Exception("FIX THIS!")


dataset = TEST_Dataset(
    data_dir=data_dir, num_samples=num_samples, transform=transform, split=split)

batch_size = 32
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
    model = FINAL_Model(video_predictor=video_predictor, video_predictor_path=video_predictor_path,
                        segmentation=segmentation, segmentation_path=segmentation_path,
                        split=split).cuda()
model.eval()
total_params = sum(p.numel() for p in model.parameters())
print(f"Number of model parameters - {total_params}")

stacked_pred = []  # stacked predicted segmentation of predicted 22nd frame
# stacked predicted segmentation of actual 22nd frame (only if not test)
stacked_target = []
stacked_gt = []  # stacked actual segmentation (only if train/val)

unique_original_objects = []

save_images = True  # set to True to save images

colors = np.random.rand(49, 3)
# Create a colormap from these colors
cmap = mcolors.ListedColormap(colors)

with torch.no_grad():
    model.eval()

    for it, (_, input_images, target_images, gt_mask) in tqdm(enumerate(dataloader)):
        input_images = input_images.cuda()  # B, T, C, H, W
        target_images = target_images.cuda()
        input_img_masks = gt_mask[:, :11].cpu()
        gt_mask = gt_mask[:, -1]

        if (video_predictor != "ft_simvp"):
            pred_mask, target_mask, pred_image = model(
                input_images, target_images)
        else:
            pred_mask = model(input_images)
            pred_mask = torch.argmax(pred_mask, dim=2)
            pred_mask = pred_mask[:, -1, :, :]

        if save_images:
            for b in range(pred_mask.shape[0]):
                vertical_line = np.ones(
                    (pred_mask.shape[1], 10), dtype=np.uint8)*50
                hstacked = np.hstack(
                    [gt_mask[b].cpu().numpy(), vertical_line, pred_mask[b].cpu().numpy()])
                plt.imsave(f"./images/{it}_{b}.png", hstacked, cmap=cmap)

        # print(pred_mask.shape, target_images.shape, gt_mask.shape)
        unique_original_objects = unique_original_objects + get_unique_objects(
            input_img_masks.cpu().numpy())

        if (video_predictor != "ft_simvp"):
            # wandb images.
            w_masks, w_pred_img = plot_images(pred_mask.detach().cpu().numpy()[0],
                                              target_mask.detach().cpu().numpy()[
                0],
                pred_image.detach().cpu().numpy()[
                0].transpose(1, 2, 0),
                target_images.detach().cpu().numpy()[0][-1].transpose(1, 2, 0))

        stacked_pred.append(pred_mask.cpu())
        if split != "test" and split != "unlabeled":
            if (video_predictor != "ft_simvp"):
                stacked_target.append(target_mask.cpu())
            stacked_gt.append(gt_mask.cpu())

    print("Number of Unique original objects: ", len(unique_original_objects))
    stacked_pred = torch.cat(stacked_pred, 0)
    print(f"Stacked Pred shape - {stacked_pred.shape}")
    if split != "test" and split != "unlabeled":
        if (video_predictor != "ft_simvp"):
            stacked_target = torch.cat(stacked_target, 0)
            print(f"Stacked Orig shape - {stacked_target.shape}")
        stacked_gt = torch.cat(stacked_gt, 0)
        print(f"Stacked GT shape - {stacked_gt.shape}")

    if split != 'test':
        jaccard = torchmetrics.JaccardIndex(task="multiclass", num_classes=49)
        jaccard_val = jaccard(stacked_pred, stacked_gt)
        confusion_mat = compute_confusion_matrix(stacked_gt, stacked_pred, 49)
        plot_confusion_matrix(confusion_mat, "Predicted Segmentation")
        torch.save(stacked_pred, "stacked_pred_no_h.pt")

        fixed_stacked_pred = apply_heuristics(stacked_pred,
                                              unique_original_objects, 'connected_components')
        jaccard_val_h = jaccard(fixed_stacked_pred, stacked_gt)
        confusion_mat = compute_confusion_matrix(
            stacked_gt, fixed_stacked_pred, 49)
        plot_confusion_matrix(
            confusion_mat, "Predicted Segmentation after Heuristics")
        print("Jaccard of predicted with gt: ", jaccard_val)
        print("Jaccard of predicted with gt after heuristics: ", jaccard_val_h)

        if (video_predictor != "ft_simvp"):
            jaccard_val = jaccard(stacked_target, stacked_gt)
            print("Jaccard of original with gt: ", jaccard_val)
            confusion_mat = compute_confusion_matrix(
                stacked_gt, stacked_target, 49)
            plot_confusion_matrix(confusion_mat, "Target Segmentation")
        jaccard_gt = jaccard(stacked_gt, stacked_gt)
        print("Jaccard of gt with gt: ", jaccard_gt)
        confusion_mat = compute_confusion_matrix(
            stacked_gt, stacked_gt, 49)
        plot_confusion_matrix(confusion_mat, "Truth")
