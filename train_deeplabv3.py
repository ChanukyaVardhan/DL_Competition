import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
import torchmetrics
from dataloader import CLEVRERSegDataset
import wandb
import numpy as np
from utils import class_labels
from models import count_parameters

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


def plot_masks(pred_mask, gt_mask, image, idx):
    # Plot the predicted mask and the ground truth mask side by side with the IoU score
    image = unnormalize(image)

    return wandb.Image(image, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": class_labels},
        "ground truth": {"mask_data": gt_mask, "class_labels": class_labels}
    })


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", default="config/segmentation_deeplabv3.yml", help="Path to config file.")
    parser.add_argument("--save_every",  default=False, action="store_true",
                        help="Flag to save every few epochs to True.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params["experiment"] = os.path.splitext(
        os.path.basename(args.config_path))[0]
    params["is_save_every"] = args.save_every

    return params


if __name__ == "__main__":

    params = get_parameters()

    # Set seed
    torch.manual_seed(params["seed"])
    torch.cuda.manual_seed_all(params["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:  # Multiple GPUs
        params["batch_size"] *= num_gpus
        params["num_workers"] *= num_gpus

    wandb.init(
        entity="dl_competition",
        config=params,
    )

    # Prepare the dataset
    # Set the appropriate arguments for your dataset class
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30),
        transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
            0.0571, 0.0567, 0.0614])
    ])
    transform_mask = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(30)
    ])
    data_dir = params["data_dir"]
    train_dataset = CLEVRERSegDataset(
        data_dir=data_dir, split='train', user_transforms=transform, mask_transform=transform_mask)
    val_dataset = CLEVRERSegDataset(
        data_dir=data_dir, split='val', user_transforms=transform, mask_transform=transform_mask, num_samples=1000)

    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"])
    val_loader = DataLoader(
        val_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])

    # Create the model
    model = deeplabv3_resnet50(
        num_classes=params["num_classes"], weights_backbone=None)
    model = nn.DataParallel(model).to(
        device) if num_gpus > 1 else model.to(device)

    # Print number of parameters
    print(f"Number of parameters: {count_parameters(model)}")
    model.to(device)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(
        model.parameters(), lr=float(params['lr']), weight_decay=float(params['weight_decay']))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', factor=0.5, verbose=True, min_lr=1e-6)

    # Training loop
    num_epochs = int(params["num_epochs"])
    min_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            train_loss += loss.item()
            loss.backward()

            optimizer.step()

        train_loss /= len(train_loader)
        wandb.log({"train_loss": train_loss})

        # Validation loop
        if epoch % 5 == 0:
            model.eval()
            eval_loss = 0
            stacked_pred = []
            stacked_gt = []
            jaccard = torchmetrics.JaccardIndex(
                task="multiclass", num_classes=params["num_classes"])
            with torch.no_grad():
                for i, (images, masks) in enumerate(val_loader):
                    images, gt_mask = images.to(device), masks.to(device)

                    outputs = model(images)['out']
                    loss = criterion(outputs, gt_mask)
                    eval_loss += loss.item()

                    # Calculate the validation metrics
                    pred_mask = torch.argmax(outputs, dim=1)
                    stacked_pred.append(pred_mask.cpu())
                    stacked_gt.append(gt_mask.cpu())

                    if i % 100 == 0:
                        mask = plot_masks(pred_mask[0].cpu().numpy(
                        ), gt_mask[0].cpu().numpy(), images[0].cpu(), i)
                        wandb.log({"val_predictions": mask})

            stacked_pred = torch.cat(stacked_pred, 0)
            stacked_gt = torch.cat(stacked_gt, 0)
            jaccard_score = jaccard(stacked_pred, stacked_gt)
            wandb.log({"jaccard_score": jaccard_score})

            eval_loss /= len(val_loader)
            wandb.log({"val_loss": eval_loss})

            scheduler.step(eval_loss)

            # Save the best evaluation loss model
            if eval_loss < min_val_loss:
                min_val_loss = eval_loss
                # FIX: model.module
                torch.save(model.state_dict(),
                           f'deeplab_v3_segmentation_model_{epoch}.pth')

    # Save the trained model
    # Access the inner model for saving

    # FIX: model.module
    torch.save(model.state_dict(), 'deeplab_v3_segmentation_model.pth')

    wandb.finish()
