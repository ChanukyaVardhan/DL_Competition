import torch
from train_seg import get_parameters, eval_epoch
from dataloader import CLEVRERSegDataset
from segmentation import SegNeXT
from torchvision import transforms
from torch.utils.data import DataLoader
import torchmetrics


def get_batch_entries(batch, device):
    input_images = batch[0].to(device)
    gt_mask = batch[1].to(device)

    return input_images, gt_mask


def eval_epoch(model, criterion, eval_loader, device, params):
    model.eval()
    eval_loss = 0.0
    num_batches = len(eval_loader)

    stacked_pred = None
    stacked_gt = None
    with torch.no_grad():
        for i, batch in enumerate(eval_loader):
            input_images, gt_mask = get_batch_entries(batch, device)
            batch_size = input_images.shape[0]

            output_mask = model(input_images)
            loss = criterion(output_mask, gt_mask)

            eval_loss += loss.item()
            pred_mask = torch.argmax(output_mask, dim=1)
            if stacked_pred is None:
                stacked_pred = pred_mask
                stacked_gt = gt_mask
            else:
                stacked_pred = torch.cat([stacked_pred, pred_mask], dim=0)
                stacked_gt = torch.cat([stacked_gt, gt_mask], dim=0)

    eval_loss /= num_batches

    return eval_loss, stacked_pred, stacked_gt


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

    transform = transforms.Compose([
        transforms.ToTensor(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomVerticalFlip(),
        # transforms.RandomResizedCrop(size = 224, scale = (0.8, 1.0), ratio = (0.8, 1.2)),
        transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
                             0.0571, 0.0567, 0.0614])
    ])
    data_dir = params["data_dir"]
    train_dataset = CLEVRERSegDataset(
        data_dir=data_dir, split='train', user_transforms=transform, num_samples=1000)
    val_dataset = CLEVRERSegDataset(
        data_dir=data_dir, split='val', user_transforms=transform, num_samples=1000)
    eval_loader = DataLoader(
        train_dataset, batch_size=params["batch_size"]*3, shuffle=False, num_workers=params["num_workers"])

    class_weights = torch.ones(params["num_classes"]).to(device)
    class_weights[0] = 0.2

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255)

    model = SegNeXT(params["num_classes"], weights=None)
    model = model.to(device)

    eval_loss, stacked_pred, stacked_gt = eval_epoch(
        model, criterion, eval_loader, device, params)

    jaccard = torchmetrics.JaccardIndex(
        task="multiclass", num_classes=49).to(device)

    jaccard_val = jaccard(stacked_pred, stacked_gt)
    print("Jaccard: ", jaccard_val)
