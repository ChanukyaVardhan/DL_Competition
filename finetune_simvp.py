from collections import OrderedDict
from PIL import Image
import os
import sys
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
import wandb

from our_OpenSTL.openstl.models import SimVP_Model
from our_OpenSTL.openstl.datasets import load_data
from our_OpenSTL.openstl.modules import ConvSC
# from our_OpenSTL.openstl.api import BaseExperiment
import torchmetrics
from train_seg import get_parameters, eval_epoch
from utils import class_labels


mean = [0.5061, 0.5045, 0.5008]
std = [0.0571, 0.0567, 0.0614]
unnormalize_transform = transforms.Compose([
    transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std]),
])
to_pil = transforms.ToPILImage()


def unnormalize(img):
    unnormalized_image = unnormalize_transform(img)
    pil_image = to_pil(unnormalized_image)

    return pil_image

def create_collage(images, width, height):
    collage = Image.new("RGB", (width, height))
    x_offset = 0
    for img in images:
        img = img.resize((width // len(images), height))
        collage.paste(img, (x_offset, 0))
        x_offset += img.width
    return collage

def plot_masks(pred_mask, gt_mask, image, idx):
    # Plot the predicted mask and the ground truth mask side by side with the IoU score
    image = unnormalize(image)

    return wandb.Image(image, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
        "ground truth": {"mask_data": gt_mask, "class_labels": class_labels}
    })


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    params = get_parameters()

    train_loader, val_loader, test_loader = load_data(
        "clevrer", params["batch_size"], params["val_batch_size"], params["num_workers"], params["data_root"], params["distributed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:  # Multiple GPUs
        params["batch_size"] *= num_gpus
        params["num_workers"] *= num_gpus

    wandb.init(
        entity="dl_competition",
        config=params,
    )

    config = {
        "in_shape": [11, 3, 160, 240],
        "hid_S": 64,
        "hid_T": 512,
        "N_S": 4,
        "N_T": 8,
        "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3,
        "num_classes": params["num_classes"],
    }
    # exp = BaseExperiment(args)

    model = SimVP_Model(**config)

    sim_vp_model_path = params["model_path"]
    model.load_state_dict(torch.load(sim_vp_model_path))
    print("SimVP model loaded from {}".format(sim_vp_model_path))
    print("SimVP model architecture: ")
#     print(model)
    print("Number of parameters: {}".format(count_parameters(model)))

    num_classes = params["num_classes"]

    # Replace the final two layers of the model to output segmentation masks
    C_hid = model.dec.readout.in_channels
    model.dec.dec[3] = ConvSC(
        C_hid, C_hid, params["spatio_kernel_dec"], upsampling=False)    # FIX: Figure out upsampling from the model?
    model.dec.readout = nn.Conv2d(C_hid, num_classes, 1)

    model = nn.DataParallel(model).to(
        device) if num_gpus > 1 else model.to(device)

    criterion = nn.CrossEntropyLoss()  # For segmentation tasks
    # You might want to use a smaller learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=params["ft_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    scaler = GradScaler()

    # Training loop
    num_epochs = params["ft_num_epochs"]
    min_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        start_time = time()

        for idx, (input_images, output_images, output_mask) in tqdm(enumerate(train_loader)):
            input_images, output_mask = input_images.to(
                device), output_mask.to(device)
            optimizer.zero_grad()

            with autocast(enabled=False):
                outputs_pred = model(input_images)
#                 print(outputs_pred.shape, output_mask.shape)
                output_pred_flat = outputs_pred.view(-1, num_classes)
                mask_flat = output_mask.view(-1)
                loss = criterion(output_pred_flat, mask_flat)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if idx % 10 == 0:
                wandb.log({"train_loss": loss.item()})

        train_loss /= len(train_loader)
        epoch_time = time() - start_time

        wandb.log({"train_loss_total": train_loss, "epoch_time": epoch_time})

        # Validation loop
        if epoch % 5 == 0:
            model.eval()
            eval_loss = 0.0
            stacked_pred = []
            stacked_gt = []
            jaccard = torchmetrics.JaccardIndex(
                task="multiclass", num_classes=params["num_classes"])
            with torch.no_grad():
                for i, (images, output, gt_masks) in tqdm(enumerate(val_loader)):
                    images, gt_masks = images.to(device), gt_masks.to(device)
                    outputs_pred = model(images)
                    output_pred_flat = outputs_pred.view(-1, num_classes)
                    mask_flat = gt_masks.view(-1)
                
                    loss = criterion(output_pred_flat, mask_flat)
                    eval_loss += loss.item()

                    pred_mask = torch.argmax(outputs_pred, dim=1)
                    stacked_pred.append(pred_mask[:,-1,:,:].cpu())
                    stacked_gt.append(gt_masks[:,-1,:,:].cpu())

                    if i % 100 == 0:
                        mask = plot_masks(pred_mask[0][-1].cpu().numpy(
                        ), gt_masks[0][-1].cpu().numpy(), output[0][-1], i)
                        wandb.log({"val_predictions": mask})

                stacked_pred = torch.cat(stacked_pred, dim=0)
                stacked_gt = torch.cat(stacked_gt, dim=0)
                jaccard_score = jaccard(stacked_pred, stacked_gt)
                eval_loss /= len(val_loader)

                wandb.log({"val_loss_total": eval_loss,
                          "jaccard_score": jaccard_score})

                scheduler.step(eval_loss)

                if eval_loss < min_val_loss:
                    min_val_loss = eval_loss
                    torch.save(model.state_dict(),
                               f'simvp_segmentation_model_{epoch}.pth')
                    print(
                        f"Model saved at epoch {epoch} with val loss: {min_val_loss}")

    # Save the trained model
    # Access the inner model for saving
    torch.save(model.module.state_dict(), 'simvp_segmentation_model.pth')

    wandb.finish()
