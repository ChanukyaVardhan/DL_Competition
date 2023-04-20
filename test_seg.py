import torch
from train_seg import get_parameters, eval_epoch
from dataloader import CLEVRERSegDataset
from segmentation import SegNeXT
from torchvision import transforms
from torch.utils.data import DataLoader

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
    val_dataset = CLEVRERSegDataset(
        data_dir=data_dir, split='val', user_transforms=transform)
    eval_loader = DataLoader(
        val_dataset, batch_size=params["batch_size"]*3, shuffle=False, num_workers=params["num_workers"])

    class_weights = torch.ones(params["num_classes"]).to(device)
    class_weights[0] = 0.2

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=255)

    model = SegNeXT(params["num_classes"], weights=None)
    model = model.to(device)

    eval_loss, mIoU = eval_epoch(
        model, criterion, eval_loader, device, params)

    print(eval_loss, mIoU)
