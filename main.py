from dataset import SampledDataset, collate_fn
from models import PreTrainModel, VICReg
from torch.utils.data import ConcatDataset, DataLoader

import argparse
import gc
import json
import numpy as np
import os
import random
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
import wandb

def get_parameters():
	parser 	= argparse.ArgumentParser()
	parser.add_argument("--config_path", default = "config/default.yml", help = "Path to config file.")
	parser.add_argument("--pretrain",    default = False, action = "store_true", help = "Flag to set pretraining to True.")
	parser.add_argument("--save_every",  default = False, action = "store_true", help = "Flag to save every few epochs to True.")
	parser.add_argument("--load_path",   default = "", help = "Override Load the model from this path in the config.")
	parser.add_argument("--num_epochs",  default = 0, type = int, help = "Number of epochs to override value in the config.")
	parser.add_argument("--batch_size",  default = 0, type = int, help = "Batch Size to override value in the config.")
	parser.add_argument("--num_workers", default = 0, type = int, help = "Num workers to override value in the config.")
	parser.add_argument("--dropout",     default = 0, type = float, help = "Dropout to override value in the config.")
	args 	= parser.parse_args()

	with open(args.config_path, "r") as f:
		params = yaml.load(f, Loader=yaml.SafeLoader)
	params["experiment"] 	= os.path.splitext(os.path.basename(args.config_path))[0]
	params["is_pretrain"]	= args.pretrain
	params["is_save_every"] = args.save_every

	if args.load_path != "":
		params["load_path"] = load_path
	if args.num_epochs != 0:
		params["num_epochs"] = args.num_epochs
	if args.batch_size != 0:
		params["batch_size"] = args.batch_size
	if args.num_workers != 0:
		params["num_workers"] = args.num_workers
	if args.dropout != 0:
		params["dropout"] = args.dropout

	return params

def get_save_paths_prefix(params):
	prefix 			= f'{params["checkpoint_path"]}/{params["experiment"]}_'
	if params["is_pretrain"]:
		prefix += "pretrain_"

	return prefix

def get_save_paths(params):
	prefix 			= get_save_paths_prefix(params)

	model_path 		= f'{prefix}model.pt'
	return model_path

def get_batch_entries(batch, device):
	input_images 	= batch["input_images"].to(device)
	input_frames 	= batch["input_frames"].to(device)
	start_frame 	= batch["start_frame"].to(device)
	pred_image 		= batch["pred_image"].to(device)
	pred_frame 		= batch["pred_frame"].to(device)
	input_mask		= batch["input_mask"].to(device)
	pred_mask		= batch["pred_mask"].to(device)

	return input_images, input_frames, start_frame, pred_image, pred_frame, input_mask, pred_mask

def train_epoch(model, optimizer, criterion, train_loader, device, params):
	model.train()
	train_loss 		= 0.0
	num_samples 	= 0

	for i, batch in enumerate(train_loader):
		input_images, input_frames, start_frame, pred_image, pred_frame, input_mask, pred_mask = get_batch_entries(batch, device)
		batch_size 		= pred_image.shape[0]

		optimizer.zero_grad()

		if params["is_pretrain"]:
			x_encoding, x_encoding_pred, y_encoding = model(input_images, input_frames, start_frame, pred_image, pred_frame)
			loss 		= criterion(x_encoding, x_encoding_pred, y_encoding)
		else:
			raise Exception("Not implemented Yet!")

		train_loss 	   += loss.item()
		# FIX THIS - COMPUTE ACCURACY HERE?

		loss.backward()
		optimizer.step()

		num_samples    += batch_size

	train_loss /= num_samples

	return train_loss

def eval_epoch(model, criterion, eval_loader, device, params):
	model.eval()
	eval_loss 		= 0.0
	num_samples 	= 0

	for i, batch in enumerate(eval_loader):
		input_images, input_frames, start_frame, pred_image, pred_frame, input_mask, pred_mask = get_batch_entries(batch, device)
		batch_size 		= pred_image.shape[0]

		if params["is_pretrain"]:
			x_encoding, x_encoding_pred, y_encoding = model(input_images, input_frames, start_frame, pred_image, pred_frame)
			loss 		= criterion(x_encoding, x_encoding_pred, y_encoding)
		else:
			raise Exception("Not implemented Yet!")

		eval_loss 	   += loss.item()
		# FIX THIS - COMPUTE ACCURACY HERE?

		num_samples    += batch_size

	eval_loss /= num_samples

	return eval_loss

def train_model(model, optimizer, criterion, train_loader, eval_loader, device, params):
	model.train()

	start_epoch 	= 1
	best_model_path = get_save_paths(params)
	best_eval_loss 	= float("inf")
	if params["resume_training"]:
		if params["load_path"] != "": # Load the model from this path, it could the best model or form some other epoch
			model_details 	= torch.load(params["load_path"])
		elif os.path.exists(best_model_path): # Load from best model path
			model_details 	= torch.load(best_model_path)
		else:
			raise Exception("Can't resume training!")

		model_details 	= torch.load(best_model_path)
		start_epoch		= model_details["epoch"] + 1 # Start from the epoch after the checkpoint
		best_eval_loss	= model_details["best_eval_loss"]
		model.load_state_dict(model_details["model"])
		optimizer.load_state_dict(model_details["optimizer"])

	for epoch in range(start_epoch, params["num_epochs"] + 1):
		print(f"Training Epoch - {epoch}")

		start_time 	= time.time()
		train_loss 	= train_epoch(model, optimizer, criterion, train_loader, device, params)
		torch.cuda.empty_cache()
		train_time 	= time.time() - start_time
		print(f"Training Loss - {train_loss:.4f}, Training Time - {train_time:.2f} secs")

		start_time 	= time.time()
		eval_loss 	= eval_epoch(model, criterion, eval_loader, device, params)
		torch.cuda.empty_cache()
		eval_time 	= time.time() - start_time
		print(f"Eval Loss - {eval_loss:.4f}, Eval Time - {eval_time:.2f} secs")

		gc.collect()

		wandb.log({"Train Loss": train_loss, "Eval Loss": eval_loss, "Train Time": train_time, "Eval Time": eval_time})

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
			prefix 		= get_save_paths_prefix(params)
			epoch_path 	= f'{prefix}model_{epoch}.pt'
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
	np.random.seed(params["seed"])
	random.seed(params["seed"])
	torch.manual_seed(params["seed"])
	torch.cuda.manual_seed_all(params["seed"])
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	device		= torch.device("cuda" if torch.cuda.is_available() else "cpu")
	num_gpus 	= torch.cuda.device_count()
	if num_gpus > 1: # Multiple GPUs
		params["batch_size"] *= num_gpus
		params["num_workers"] *= num_gpus

	wandb.init(
		entity = "dl_competition",
		config = params,
	)

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		# FIX WITH DATA AUGMENTATIONS
		transforms.ToTensor(),
		transforms.Normalize(mean = [0.5061, 0.5045, 0.5008], std = [0.0571, 0.0567, 0.0614])
	])

	# Datasets
	# FIX THIS - SHOULD THE DATASET BE THE WHOLE SAMPLED DATASET OR JUST THE VIDEOS?
	if params["is_pretrain"]:
		train_dataset 	= ConcatDataset([
							SampledDataset(data_dir = params["data_dir"], split = "unlabeled", transform = transform),
							SampledDataset(data_dir = params["data_dir"], split = "train", transform = transform),
						])	
		eval_dataset 	= SampledDataset(data_dir = params["data_dir"], split = "val", transform = transform)
	else:
		raise Exception("Not implemented Yet!")

	# Dataloaders
	if params["is_pretrain"]:
		train_loader 	= DataLoader(train_dataset, batch_size = params["batch_size"], shuffle = True, collate_fn = collate_fn, num_workers = params["num_workers"])
		eval_loader 	= DataLoader(eval_dataset, batch_size = params["batch_size"], shuffle = False, collate_fn = collate_fn, num_workers = params["num_workers"])
	else:
		raise Exception("Not implemented Yet!")

	# Model
	if params["is_pretrain"]:
		model = PreTrainModel(
					cnn_encoder = params["cnn_encoder"],
					d_emb 		= params["d_emb"],
					d_ff 		= params["d_ff"],
					n_heads 	= params["n_heads"],
					n_layers 	= params["n_layers"],
					dropout 	= params["dropout"],
				)
	else:
		raise Exception("Not implemented Yet!")
	# model = model.to(device)
	model = nn.DataParallel(model).to(device) if num_gpus > 1 else model.to(device)

	trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"Number of model parameters - {trainable_params}")

	# Optimizer
	if params["optimizer"] == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr = params["lr"])
	else:
		raise Exception(f'Optimizer {params["optimizer"]} is not supported!')

	# Loss
	if params["is_pretrain"]:
		criterion = VICReg(params["sim_coeff"], params["std_coeff"], params["cov_coeff"]).to(device)
	else:
		raise Exception("Not implemented Yet!")

	# Training
	train_model(model, optimizer, criterion, train_loader, eval_loader, device, params)

	wandb.finish()
