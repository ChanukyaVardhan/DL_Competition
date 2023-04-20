import argparse
import yaml
import os
import torch
import torchmetrics
import time
import gc
from tqdm import tqdm

from torch.utils.data import DataLoader

from torchvision import transforms

from dataloader import CLEVRERSegDataset
from segmentation import SegNeXT
from main import get_save_paths_prefix, get_save_paths
import matplotlib.pyplot as plt
import wandb


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", default="config/segmentation_default.yml", help="Path to config file.")
    parser.add_argument("--save_every",  default=False, action="store_true",
                        help="Flag to save every few epochs to True.")
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)
    params["experiment"] = os.path.splitext(
        os.path.basename(args.config_path))[0]
    params["is_save_every"] = args.save_every

    return params


def get_batch_entries(batch, device):
    input_images = batch[0].to(device)
    gt_mask = batch[1].to(device)

    return input_images, gt_mask


def train_epoch(model, optimizer, criterion, train_loader, device, params):
    model.train()
    train_loss = 0.0

    for i, batch in tqdm(enumerate(train_loader)):
        input_images, gt_mask = get_batch_entries(batch, device)

        optimizer.zero_grad()

        output_mask = model(input_images)
#         print(gt_mask.dtype, output_mask.dtype)
        loss = criterion(output_mask, gt_mask)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)

    return train_loss


def plot_masks(pred_mask, gt_mask, iou_label, idx):
    # Plot the predicted mask and the ground truth mask side by side with the IoU score
    fig, ax = plt.subplots(1, 2, figsize=(15, 15))
    ax[0].imshow(pred_mask)
    ax[0].set_title("Predicted Mask")
    ax[1].imshow(gt_mask)
    ax[1].set_title("Ground Truth Mask")
    plt.suptitle("IoU: {}".format(iou_label))
    plt.savefig(f"result_{idx}.png")


def eval_epoch(model, criterion, eval_loader, device, params):
    model.eval()
    eval_loss = 0.0

    mIoU = 0.0
    with torch.no_grad():
        jaccard = torchmetrics.JaccardIndex(
            task="multiclass", num_classes=49, ignore_index=255).to(device)
        for i, batch in tqdm(enumerate(eval_loader)):
            input_images, gt_mask = get_batch_entries(batch, device)
            batch_size = input_images.shape[0]

            output_mask = model(input_images)
            loss = criterion(output_mask, gt_mask)

            eval_loss += loss.item()
            # COMPUTE mIoU
            pred_mask = torch.softmax(output_mask, dim=1)
            pred_mask = torch.argmax(pred_mask, dim=1)
            print(pred_mask[0])
            pred_mask = torch.argmax(output_mask, dim=1)
            print(torch.unique(pred_mask[0], return_counts=True))
            plot_masks(pred_mask[0].cpu().numpy(),
                       gt_mask[0].cpu().numpy(), 0, i)
            iou = jaccard(pred_mask, gt_mask)
            print(pred_mask.shape, gt_mask.shape, iou)
            mIoU += jaccard(pred_mask, gt_mask)
            

    eval_loss /= len(eval_loader)
    mIoU /= len(eval_loader)

    return eval_loss, mIoU


def train_model(model, optimizer, criterion, train_loader, eval_loader, device, params):
    model.train()

    start_epoch = 1
    best_model_path = get_save_paths(params)
    best_eval_loss = float("inf")
    if params["resume_training"]:
        # Load the model from this path, it could the best model or form some other epoch
        if params["load_path"] != "":
            load_path = params["load_path"]
        elif os.path.exists(best_model_path):  # Load from best model path
            load_path = params[best_model_path]
        else:
            raise Exception("Can't resume training!")

        print(f"Loading model from - {load_path}")
        model_details = torch.load(load_path)
        # Start from the epoch after the checkpoint
        start_epoch = model_details["epoch"] + 1
        best_eval_loss = model_details["best_eval_loss"]
        model.load_state_dict(model_details["model"])
        optimizer.load_state_dict(model_details["optimizer"])

    for epoch in range(start_epoch, params["num_epochs"] + 1):
        print(f"Training Epoch - {epoch}")

        start_time = time.time()
        train_loss = train_epoch(
            model, optimizer, criterion, train_loader, device, params)
        torch.cuda.empty_cache()
        train_time = time.time() - start_time
        print(
            f"Training Loss - {train_loss:.4f}, Training Time - {train_time:.2f} secs")

        wandb.log({"Train Loss": train_loss, "Train Time": train_time})

        start_time = time.time()
        eval_loss, mIoU = eval_epoch(
            model, criterion, eval_loader, device, params)
        torch.cuda.empty_cache()
        eval_time = time.time() - start_time
        print(f"Eval Loss - {eval_loss:.4f}, Eval Time - {eval_time:.2f} secs")

        wandb.log({"Eval Loss": eval_loss, "Eval Time": eval_time, "mIoU": mIoU})

        gc.collect()

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            print(f"Saving model with best eval loss - {best_eval_loss:.4f}")
            torch.save({
                "epoch": 			epoch,
                "best_eval_loss":	best_eval_loss,
                "model": 			model.state_dict(),
                "optimizer":		optimizer.state_dict()
            }, best_model_path)

        # Save model after every few epochs
        if params["is_save_every"] and (epoch % params["save_every"] == 0):
            prefix = get_save_paths_prefix(params)
            epoch_path = f'{prefix}model_{epoch}.pt'
            print(f"Saving model at epoch - {epoch}")
            torch.save({
                "epoch": 			epoch,
                "best_eval_loss":	best_eval_loss,
                "model": 			model.state_dict(),
                "optimizer":		optimizer.state_dict()
            }, epoch_path)

    return model


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

    transform = transforms.Compose([
        transforms.ToTensor(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        # transforms.RandomResizedCrop(size = 224, scale = (0.8, 1.0), ratio = (0.8, 1.2)),
        transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
                             0.0571, 0.0567, 0.0614])
    ])

    # Create the datasets and dataloaders
    data_dir = params["data_dir"]
    train_dataset = CLEVRERSegDataset(
        data_dir=data_dir, split='train', user_transforms=transform)
    val_dataset = CLEVRERSegDataset(
        data_dir=data_dir, split='val', user_transforms=transform)

    print(len(train_dataset), len(val_dataset))
    train_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"], shuffle=True, num_workers=params["num_workers"])
    eval_loader = DataLoader(
        val_dataset, batch_size=params["batch_size"]*3, shuffle=False, num_workers=params["num_workers"])

    model = SegNeXT(params["num_classes"], weights=None)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["lr"]))
    # Define class weights. Les weight for background class. Total 49 classes where 0 is background
    class_weights = torch.ones(params["num_classes"]).to(device)
    class_weights[0] = 0.2

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255)

    train_model(model, optimizer, criterion,
                train_loader, eval_loader, device, params)

    wandb.finish()
