from collections import OrderedDict
from PIL import Image
import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from our_OpenSTL.openstl.models import SimVP_Model
from our_OpenSTL.openstl.datasets import load_data
from our_OpenSTL.openstl.modules import ConvSC
# from our_OpenSTL.openstl.api import BaseExperiment
import torchmetrics
from train_seg import get_parameters, eval_epoch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    params = get_parameters()

    train_loader, val_loader, test_loader = load_data(
        "clevrer", params["batch_size"], params["val_batch_size"], params["num_workers"], params["data_root"], params["distributed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = {
        "in_shape": [11, 3, 160, 240],
        "hid_S": 64,
        "hid_T": 512,
        "N_S": 4,
        "N_T": 8,
        "spatio_kernel_enc": 3,
        "spatio_kernel_dec": 3
    }
    # exp = BaseExperiment(args)

    model = SimVP_Model(**config)
    sim_vp_model_path = params["model_path"]
    model.load_state_dict(torch.load(sim_vp_model_path))
    print("SimVP model loaded from {}".format(sim_vp_model_path))
    print("SimVP model architecture: ")
    print(model)
    print("Number of parameters: {}".format(count_parameters(model)))

    num_classes = params["num_classes"]

    # Replace the final two layers of the model to output segmentation masks
    C_hid = model.dec.readout.in_channels
    model.dec[3] = ConvSC(
        C_hid, C_hid, params["spatio_kernel_dec"], upsampling=False)    # Figure out upsampling from the model?
    model.dec.readout = nn.Conv2d(C_hid, num_classes, 1)

    criterion = nn.CrossEntropyLoss()  # For segmentation tasks
    # You might want to use a smaller learning rate for fine-tuning
    optimizer = torch.optim.Adam(model.parameters(), lr=params["ft_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Training loop
    num_epochs = params["ft_num_epochs"]
