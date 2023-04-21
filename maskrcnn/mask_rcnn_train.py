import random
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import os
import torch.utils.data
import torchvision.models.segmentation
import torchvision.transforms as transforms
import torch
from dataset import MaskRCNNDataset, mrcnn_collate_fn
from torch.utils.data import DataLoader
import wandb
import gc
import time
import argparse


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path",   default="",
                        help="Override Load the model from this path in the config.")
    parser.add_argument("--data_dir",   default=".",
                        help="Directory where data is stored. Train / Val / Test should be in this directory.")
    parser.add_argument("--num_epochs",  default=0, type=int,
                        help="Number of epochs to override value in the config.")
    parser.add_argument("--batch_size",  default=0, type=int,
                        help="Batch Size to override value in the config.")
    parser.add_argument("--num_workers", default=0, type=int,
                        help="Num workers to override value in the config.")
    parser.add_argument("--dropout",     default=0, type=float,
                        help="Dropout to override value in the config.")
    parser.add_argument("--model_save_path",    default=".",
                        help="Directory where model will be saved.")
    parser.add_argument("--learning_rate",     default=1e-5,
                        type=float, help="Learning rate")

    args = parser.parse_args()

    params = {}
    params["experiment"] = "maskrcnn_" + str(args.num_epochs) + \
        "_" + str(args.batch_size) + "___" + str(args.learning_rate)
    params["model_save_path"] = args.model_save_path

    if args.load_path != "":
        params["load_path"] = args.load_path
    params["data_dir"] = args.data_dir
    assert (args.num_epochs > 0 and args.batch_size > 0)
    params["num_epochs"] = args.num_epochs
    params["batch_size"] = args.batch_size
    params["num_workers"] = args.num_workers
    if args.dropout != 0:
        params["dropout"] = args.dropout
    params["learning_rate"] = args.learning_rate

    return params


def train_model(model, train_loader, optimizer, device, epoch):
    total_loss = 0.0
    total_batches = 0
    for i, batch in enumerate(train_loader):
        # print(batch)
        images, targets = batch
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        optimizer.zero_grad()
        outputs = model(images, targets)
        loss = sum(loss for loss in outputs.values())
        loss.sum().backward()
        optimizer.step()
        total_loss += loss.sum().item() / images.size(0)
        total_batches += 1

    print("Epoch: ", epoch, "Train Loss: ", total_loss / total_batches)

    return total_loss / total_batches


def eval_model(model, eval_loader, device, epoch):
    total_loss = 0.0
    total_batches = 0
    for i, batch in enumerate(eval_loader):
        images, targets = batch
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images, targets)
        loss = sum(loss for loss in outputs.values())
        total_loss += loss.sum().item() / images.size(0)
        total_batches += 1

    print("Epoch: ", epoch, "Eval Loss: ", total_loss / total_batches)
    return total_loss / total_batches


if __name__ == "__main__":

    params = get_parameters()
    batch_size = params["batch_size"]
    num_workers = params["num_workers"]

    imageSize = [160, 240]
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Parallel training.
    num_gpus = torch.cuda.device_count()
    print("Number of GPUs: ", num_gpus)
    if num_gpus > 1:
        batch_size = batch_size * num_gpus
        num_workers = num_workers * num_gpus

    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes=49)  # replace the pre-trained head with a new one
    model = torch.nn.DataParallel(model).to(
        device) if num_gpus > 1 else model.to(device)

    params_with_grad = filter(lambda p: p.requires_grad, model.parameters())
    params_count = sum([np.prod(p.size()) for p in params_with_grad])
    print(f"Total number of trainable parameters: {params_count}")

    num_epochs = params["num_epochs"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
                             0.0571, 0.0567, 0.0614])
    ])

    wandb.init(
        entity="dl_competition",
        config={"epochs": num_epochs,
                "batch_size": batch_size, "learning_rate": params["learning_rate"]},
    )

    # load the train dataset
    # FIX - Does it require the transforms?
    train_dataset = MaskRCNNDataset(
        data_dir=params["data_dir"], split='train', transforms=transform)
    eval_dataset = MaskRCNNDataset(
        data_dir=params["data_dir"], split='val', transforms=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=mrcnn_collate_fn, num_workers=num_workers)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size,
                             shuffle=False, collate_fn=mrcnn_collate_fn, num_workers=num_workers)

    optimizer = torch.optim.AdamW(
        params=model.parameters(), lr=params["learning_rate"])

    model_save_dir = f'{params["model_save_path"]}/{params["experiment"]}/'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    minm_loss = 100000000.0

    for epoch in range(num_epochs):
        start_time  = time.time()
        train_loss = train_model(model, train_loader, optimizer, device, epoch)
        epoch_time = time.time() - start_time
        torch.cuda.empty_cache()
        loss = eval_model(model, eval_loader, device, epoch)
        torch.cuda.empty_cache()
        wandb.log({"train_loss": train_loss, "eval_loss": loss, "epoch_time": epoch_time})
        gc.collect()
        if loss < minm_loss:
            minm_loss = loss
            torch.save({
                "epoch": 			epoch,
                "best_eval_loss":	loss,
                "model": 			model.state_dict(),
                "optimizer":		optimizer.state_dict()
            }, model_save_dir + "best_model.pt")
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": 			epoch,
                "best_eval_loss":	loss,
                "model": 			model.state_dict(),
                "optimizer":		optimizer.state_dict()
            }, model_save_dir + "model_" + str(epoch) + ".pt")

    wandb.finish()
