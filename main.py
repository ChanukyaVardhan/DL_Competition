from dataset import SampledDataset, collate_fn
from models import PreTrainModel, VICReg
from torch.utils.data import ConcatDataset, DataLoader

import argparse
import json
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml

def get_parameters():
	parser 	= argparse.ArgumentParser()
	parser.add_argument("--config_path", default = "config/default.yml", help = "Path to config file.")
	args 	= parser.parse_args()

	with open(args.config_path, "r") as f:
		params = yaml.load(f, Loader=yaml.SafeLoader)
	params["experiment"] = os.path.splitext(os.path.basename(args.config_path))[0]

	return params

def get_save_paths(params):
	prefix 			= f'{params["checkpoint_path"]}/{params["experiment"]}_'
	if params["is_pretrain"]:
		prefix += "pretrain_"
	model_path 		= f'{prefix}model.pt'
	train_stat_path = f'{prefix}stats.json'

	return model_path, train_stat_path

def get_existing_stats(train_stat_path, start_epoch, params):
	train_stats = {
		"epoch": 		[],
		"train_loss": 	[],
		"eval_loss":	[],
		# FIX THIS - ADD OTHER METRICS?
	}

	if params["resume_training"] and os.path.exists(train_stat_path):
		existing_stats = json.load(open(train_stat_path, "r"))

		for key, val in existing_stats.items():
			if key in train_stats:
				train_stats[key] = val[:start_epoch - 1]

	return train_stats

def train_epoch(model, optimizer, criterion, train_loader, device, params):
	model.train()
	train_loss 		= 0.0
	num_samples 	= 0

	for i, batch in enumerate(train_loader):
		input_images 	= batch["input_images"]
		input_frames 	= batch["input_frames"]
		pred_image 		= batch["pred_image"]
		pred_frame 		= batch["pred_frame"]
		if not params["is_pretrain"]:
			input_mask	= batch["input_mask"]
			pred_mask	= batch["pred_mask"]
		batch_size 		= pred_image.shape[0]

		optimizer.zero_grad()

		if params["is_pretrain"]:
			x_encoding, y_encoding = model(input_images, input_frames, pred_image, pred_frame)
			loss 		= criterion(x_encoding, y_encoding)
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
		input_images 	= batch["input_images"]
		input_frames 	= batch["input_frames"]
		pred_image 		= batch["pred_image"]
		pred_frame 		= batch["pred_frame"]
		if not params["is_pretrain"]:
			input_mask	= batch["input_mask"]
			pred_mask	= batch["pred_mask"]
		batch_size 		= pred_image.shape[0]

		if params["is_pretrain"]:
			x_encoding, y_encoding = model(input_images, input_frames, pred_image, pred_frame)
			loss 		= criterion(x_encoding, y_encoding)
		else:
			raise Exception("Not implemented Yet!")

		eval_loss 	   += loss.item()
		# FIX THIS - COMPUTE ACCURACY HERE?

		num_samples    += batch_size

	eval_loss /= num_samples

	return eval_loss

def train_model(model, optimizer, criterion, train_loader, eval_loader, device, params):
	model.train()

	start_epoch = 1
	model_path, train_stat_path = get_save_paths(params)
	if params["resume_training"] and os.path.exists(model_path):
		model_details 	= torch.load(model_path)
		start_epoch		= model_details["epoch"] + 1 # Start from the epoch after the checkpoint
		model.load_state_dict(model_details["model"])
		optimizer.load_state_dict(model_details["optimizer"])

	train_stats = get_existing_stats(train_stat_path, start_epoch, params)

	for epoch in range(start_epoch, params["num_epochs"] + 1):
		print(f"Training Epoch - {epoch}")

		train_loss 	= train_epoch(model, optimizer, criterion, train_loader, device, params)
		eval_loss 	= eval_epoch(model, criterion, eval_loader, device, params)

		print(f"Training Loss - {train_loss:.4f}, Eval Loss - {eval_loss:.4f}")
		
		train_stats["epoch"].append(epoch)
		train_stats["train_loss"].append(train_loss)
		train_stats["eval_loss"].append(eval_loss)

		with open(train_stat_path, "w") as f:
			json.dump(train_stats, f)

		# FIX THIS - SAVE MODEL AND OPTIMIZER ON SOME CONDITION, ALSO SAVE THE CONDITION IN THE PATH AS WELL
		torch.save({
			"epoch": 		epoch,
			"model": 		model.state_dict(),
			"optimizer":	optimizer.state_dict()
		}, model_path)

	return model

if __name__ == "__main__":
	params = get_parameters()

	# Set seed
	torch.manual_seed(params["seed"])
	torch.cuda.manual_seed_all(params["seed"])
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	device	= torch.device("cuda" if torch.cuda.is_available() else "cpu")

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		# FIX WITH DATA AUGMENTATIONS
		transforms.ToTensor(),
		# FIX WITH APPROPRIATE MEAN AND STD VALUES
		# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
					dropout 	= params["dropout"]
				)
	else:
		raise Exception("Not implemented Yet!")
	model = model.to(device)

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
