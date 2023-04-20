from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.ops import masks_to_boxes

import numpy as np
import os
import torch
import torchvision.transforms as transforms

class SampledDataset(Dataset):

	def __init__(self, data_dir = "./data", split = "unlabeled", start_frame = 0, sample_frames = 11, distance = 11, transform = transforms.ToTensor()):
		self.data_dir 		= data_dir
		self.split 			= split 		# train/unlabeled/val/test
		self.path 			= os.path.join(self.data_dir, self.split)
		self.start_frame	= start_frame	# If not test split, then we sample this value from [0, 10]
		self.sample_frames	= sample_frames # FIX THIS - FOR NOW ALWAYS USING 11 FRAMES
		self.distance 		= distance 		# If not test split, then we sample this value between [1, 11]
		self.transform 		= transform

		self.video_ids 		= sorted([v for v in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, v))])
		# FIX - UNCOMMENT THIS TO RUN LOCALLY
		# self.video_ids 		= self.video_ids[:2]

	def __len__(self):
		return len(self.video_ids)

	def _load_image(self, image_path):
		image 			= Image.open(image_path)
		image 			= self.transform(image) if self.transform is not None else image

		return image

	def __getitem__(self, index):
		video_id 		= self.video_ids[index]
		video_path 		= os.path.join(self.path, video_id)
		mask_path 		= os.path.join(video_path, "mask.npy")

		# Sample start frame from the first 11 frames if not test or eval
		if self.split != 'test' or self.split != 'val':
			start_index	= np.random.randint(0, 11)
		else:
			start_index = self.start_frame

		# Get consecutive sample_frames number of frames
		input_images 	= []
		input_frames 	= []
		for index in range(start_index, start_index + self.sample_frames):
			image 		= self._load_image(os.path.join(video_path, f"image_{index}.png"))
			input_images.append(image)
			input_frames.append(index)
		input_images 	= torch.stack(input_images, dim = 0)
		input_frames 	= torch.tensor(input_frames)

		# Sample the frame to predict at some distance if not test or eval
		end_index 		= start_index + (self.sample_frames - 1)
		if self.split != 'test' or self.split != 'val':
			pred_index	= np.random.randint(end_index + 1, 22)
			pred_dist	= pred_index - end_index
		else:
			pred_dist	= self.distance
			pred_index	= end_index + pred_dist

		pred_image 		= self._load_image(os.path.join(video_path, f"image_{pred_index}.png"))
		pred_frame		= torch.tensor([pred_index])

		# Extract input and prediction mask
		mask 			= torch.FloatTensor(np.load(mask_path)) if os.path.exists(mask_path) else \
							torch.zeros((22, 160, 240))
							# torch.zeros((22, input_images.shape[2], input_images.shape[3]))
							# FIX THIS - SHOULD WE RESIZE THE MASK TO 224,224? DON'T THINK SO. HARDCODING THE VALUE FOR NOW
		input_mask 		= mask[input_frames]
		pred_mask 		= mask[pred_frame]

		instance 		= {
			"video_id": 	video_id,									# Video Id

			"input_images": input_images.unsqueeze(0),					# Input x images
			"input_frames": input_frames.unsqueeze(0),					# Frame indexes of the input x frames
			"start_frame": 	torch.tensor([start_index]).unsqueeze(0),	# Start frame index of the input x
			"input_mask":	input_mask.unsqueeze(0),					# Segmentation mask of the input x

			"pred_image": 	pred_image.unsqueeze(0),					# Image y
			"pred_frame": 	pred_frame.unsqueeze(0),					# Frame index of y
			"pred_dist":	torch.tensor([pred_dist]).unsqueeze(0),		# Distance of y from the end of x frames that we are predicting
			"pred_mask": 	pred_mask.unsqueeze(0),						# Segmentation mask of y
		}

		return instance

def collate_fn(data):
	tensor_items 	= ["input_images", "input_frames", "start_frame", "input_mask", "pred_image", "pred_frame", "pred_dist", "pred_mask"]
	batch 			= {k: [d[k] for d in data] for k in data[0].keys()}

	if len(data) == 1:
		for k,v in batch.items():
			if k in tensor_items:
				batch[k] = torch.cat(batch[k], 0)
			else:
				batch[k] = batch[k][0]
	else:
		for k in tensor_items:
			batch[k] = torch.cat(batch[k], 0)

	return batch


class MaskedDataset(Dataset): # Dataset to be used for training the final segmentation model.

	def __init__(self, data_dir = "./data", split = "train", transform = transforms.ToTensor()):
		self.data_dir 		= data_dir
		self.split 			= split 		# train/unlabeled/val/test
		self.path 			= os.path.join(self.data_dir, self.split)
		self.start_frame	= 0
		self.sample_frames	= 11
		self.distance 		= 11
		self.transform 		= transform

		self.video_ids 		= sorted([v for v in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, v))])
		# FIX - UNCOMMENT THIS TO RUN LOCALLY
		# self.video_ids 		= self.video_ids[:2]
		if (self.split == "test"):
			raise Exception("Test split not implemented yet")

	def __len__(self):
		return len(self.video_ids)

	def _load_image(self, image_path):
		image 			= Image.open(image_path)
		image 			= self.transform(image) if self.transform is not None else image

		return image
	
	def _get_bounding_boxes(self, mask):
		# Get bounding boxes for each object in the mask
		# Input: mask - (H, W) tensor of values in [0, 48]
		# Returns a tensor of bounding boxes of shape [N, x1, y1, x2, y2]
		# where (x1, y1) is the top left corner and (x2, y2) is the bottom right corner
		# of the bounding box and N is the number of objects present in the mask. N <= 48.
		obj_ids = torch.unique(mask)

		# first id is the background, so remove it.
		obj_ids = obj_ids[1:]

		masks = mask == obj_ids[:, None, None]

		boxes = masks_to_boxes(masks)
		return boxes

	def __getitem__(self, index):
		video_id 		= self.video_ids[index]
		video_path 		= os.path.join(self.path, video_id)
		mask_path 		= os.path.join(video_path, "mask.npy")

		start_index = self.start_frame

		# Get consecutive sample_frames number of frames
		input_images 	= []
		input_frames 	= []
		for index in range(start_index, start_index + self.sample_frames):
			image 		= self._load_image(os.path.join(video_path, f"image_{index}.png"))
			input_images.append(image)
			input_frames.append(index)
		input_images 	= torch.stack(input_images, dim = 0)
		input_frames 	= torch.tensor(input_frames)

		end_index 		= start_index + (self.sample_frames - 1)
		pred_dist	= self.distance
		pred_index	= end_index + pred_dist
			
		
		pred_image 		= self._load_image(os.path.join(video_path, f"image_{pred_index}.png"))
		pred_frame		= torch.tensor([pred_index])
			

		# Extract input and prediction mask
		mask 			= torch.FloatTensor(np.load(mask_path)) if os.path.exists(mask_path) else \
							torch.zeros((22, 160, 240))
							# torch.zeros((22, input_images.shape[2], input_images.shape[3]))
							# FIX THIS - SHOULD WE RESIZE THE MASK TO 224,224? DON'T THINK SO. HARDCODING THE VALUE FOR NOW
		input_mask 		= mask[input_frames]
		pred_mask 		= mask[pred_frame]

		bouding_boxes = _get_bounding_boxes(pred_mask)

		instance 		= {
			"video_id": 	video_id,									# Video Id

			"input_images": input_images.unsqueeze(0),					# Input x images
			"input_frames": input_frames.unsqueeze(0),					# Frame indexes of the input x frames
			"start_frame": 	torch.tensor([start_index]).unsqueeze(0),	# Start frame index of the input x
			"input_mask":	input_mask.unsqueeze(0),					# Segmentation mask of the input x

			"pred_image": 	pred_image.unsqueeze(0),					# Image y
			"pred_frame": 	pred_frame.unsqueeze(0),					# Frame index of y
			"pred_dist":	torch.tensor([pred_dist]).unsqueeze(0),		# Distance of y from the end of x frames that we are predicting
			"pred_mask": 	pred_mask.unsqueeze(0),						# Segmentation mask of y
			"bounding_boxes": bounding_boxes							# Bounding boxes of the objects in the prediction mask
		}

		return instance


def collate_mask_fn(data): # Same as collate_fn but for MaskedDataset
	tensor_items 	= ["input_images", "input_frames", "start_frame", "input_mask", "pred_image", "pred_frame", "pred_dist", "pred_mask", "bounding_boxes"]
	batch 			= {k: [d[k] for d in data] for k in data[0].keys()}

	if len(data) == 1:
		for k,v in batch.items():
			if k in tensor_items:
				batch[k] = torch.cat(batch[k], 0)
			else:
				batch[k] = batch[k][0]
	else:
		for k in tensor_items:
			batch[k] = torch.cat(batch[k], 0)

	return batch