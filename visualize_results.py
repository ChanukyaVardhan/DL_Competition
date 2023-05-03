import torch
import os
import numpy as np
import torchmetrics
from tqdm import tqdm
from utils import apply_heuristics, get_unique_objects
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from matplotlib import pyplot as plt
from utils import class_labels

np.random.seed(42)
colors = np.random.rand(49, 3)
# Create a colormap from these colors
cmap = mcolors.ListedColormap(colors)


def get_legend(class_indices):
    # Create a list of patches for the legend
    patches = [mpatches.Patch(color=cmap(i / (49 - 1)),
                              label=class_labels[class_indices[i]]) for i in range(len(class_indices))]

    # Return the patches
    return patches


def plot_images(prediction_mask1, prediction_mask2, save_path, class_indices1, class_indices2):
    # Create the legends
    patches1 = get_legend(class_indices1)
    patches2 = get_legend(class_indices2)

    # Create a figure and axes
    # Adjust the figure size if needed
    with plt.style.context('dark_background'):

        fig, axs = plt.subplots(2, 1, figsize=(10, 20))

        # Plot the first image
        im1 = axs[0].imshow(prediction_mask1, cmap=cmap)
        axs[0].set_title('Prediction Mask')

        # Plot the second image
        im2 = axs[1].imshow(prediction_mask2, cmap=cmap)
        axs[1].set_title('Prediction Mask After Heuristics')

        # Add the colorbars to the images
        # fig.colorbar(im1, ax=axs[0])
        # fig.colorbar(im2, ax=axs[1])

        # Add the legends to the corresponding subplots
        axs[0].legend(handles=patches1, loc='best')
        axs[1].legend(handles=patches2, loc='best')

        # Adjust the space between the plots
        # Adjust this value to get desired space
        plt.subplots_adjust(hspace=0.2)

        # Save the image with the legend
        plt.savefig(save_path)

        plt.close()


hidden_data_path = "/mnt/d/Downloads/hidden_masks/hidden"
tensor_paths = ["/mnt/d/Downloads/stacked_pred_hidden_heur.pt",
                "/mnt/d/Downloads/stacked_pred_hidden.pt"]


stacked_pred_heur = torch.load(tensor_paths[0])
stacked_pred = torch.load(tensor_paths[1])

print("Predictions: ", stacked_pred.shape)

# Read all masks from the hidden data path
video_masks = []
for idx, video in tqdm(enumerate(os.listdir(hidden_data_path))):
    if video != "video_15011":
        continue
    prediction_mask = stacked_pred[idx]
    prediction_mask_heur = stacked_pred_heur[idx]
    pred_path = os.path.join(
        hidden_data_path, video, "pred.png")
    unique_objects = np.unique(prediction_mask)
    unique_objects_heur = np.unique(prediction_mask_heur)
    # create a color map from unique objects to colors and save the image
    plot_images(prediction_mask, prediction_mask_heur,
                pred_path, unique_objects, unique_objects_heur)
