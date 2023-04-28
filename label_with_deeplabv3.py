from tqdm import tqdm
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from train_deeplabv3 import count_parameters, get_parameters
from dataloader import CLEVRERVideoSeg
from matplotlib import pyplot as plt

params = get_parameters()

# Set seed
torch.manual_seed(params["seed"])
torch.cuda.manual_seed_all(params["seed"])
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")    # To run on my stupid laptop

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5061, 0.5045, 0.5008], std=[
        0.0571, 0.0567, 0.0614])
])


data_dir = params["data_dir"]
unlabeled_dataset = CLEVRERVideoSeg(
    data_dir=data_dir, split='hidden', transform=transform)

test_loader = DataLoader(
    unlabeled_dataset, batch_size=params["batch_size"], shuffle=False, num_workers=params["num_workers"])

# Create the model
model = deeplabv3_resnet50(
    num_classes=params["num_classes"], weights_backbone=None)
# Load the checkpoint
model.load_state_dict(torch.load(
    params["test_checkpoint"], map_location='cpu'))
model.to(device)
model.eval()

with torch.no_grad():
    for idx, (input_images, mask_paths) in tqdm(enumerate(test_loader)):
        input_images = input_images.to(device)  # B x 22 x 3 x 160 x 240
        B, T, C, H, W = input_images.shape
        input_images = input_images.reshape(B*T, C, H, W)
        output_mask = model(input_images)['out']  # B*22 x C x 160 x 240
        output_mask = torch.argmax(output_mask, dim=1)  # B*22 x 160 x 240
        output_mask = output_mask.reshape(B, T, H, W)  # B x 22 x 160 x 240

        # Save the masks for each video
        for b_id in range(B):
            mask_path = mask_paths[b_id]
            mask_dir = os.path.dirname(mask_path)
            if not os.path.exists(mask_dir):
                raise ValueError(
                    f'{mask_dir} does not exist to save the masks')
            mask_tensor = output_mask[b_id].squeeze_().byte().cpu().numpy()

            # # save the masks as images
            # # Generate 49 random colors
            # colors = np.random.rand(49, 3)
            # import matplotlib.colors as mcolors
            # # Create a colormap from these colors
            # cmap = mcolors.ListedColormap(colors)
            # for t in range(T):
            #     mask_img = mask_tensor[t]
            #     mask_path = os.path.join(mask_dir, f'mask_{t}.png')
            #     # Use matplotlib to save the image
            #     plt.imsave(mask_path, mask_img, cmap=cmap)
            #     unique_labels = np.unique(mask_tensor[t])
            #     print(f"unique_labels in mask_{t}", unique_labels)
            #     from utils import get_class_name
            #     for label in unique_labels:
            #         class_name = get_class_name(label)
            #         print(f' mask_{t} {class_name}: {label}')

            np.save(mask_path, mask_tensor)

        break
