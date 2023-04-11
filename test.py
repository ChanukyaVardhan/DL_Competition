from dataset import *
from models import *
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		# FIX WITH APPROPRIATE MEAN AND STD VALUES
		# transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

unlabeled_ds = SampledDataset(data_dir = "./data", split = "unlabeled", start_frame = 0, sample_frames = 11, distance = 11, transform = transform)
unlabeled_loader = DataLoader(unlabeled_ds, collate_fn = collate_fn, batch_size = 2, num_workers = 2)

batch = next(iter(unlabeled_loader))

video_id = batch["video_id"]
input_images = batch["input_images"]
input_frames = batch["input_frames"]
pred_image = batch["pred_image"]
pred_frame = batch["pred_frame"]

# pretrainmodel = PreTrainModel(cnn_encoder = "resnet50", d_emb = 2048, d_ff = 2048, n_heads = 2, n_layers = 3, dropout = 0.)
# pretrainmodel = PreTrainModel(cnn_encoder = "resnet34", d_emb = 512, d_ff = 2048, n_heads = 2, n_layers = 3, dropout = 0.)
pretrainmodel = PreTrainModel(cnn_encoder = "resnet18", d_emb = 512, d_ff = 2048, n_heads = 2, n_layers = 3, dropout = 0.)
x_encoding, y_encoding = pretrainmodel(input_images, input_frames, pred_image, pred_frame)
print("x_encoding - ", x_encoding.shape)
print("y_encoding - ", y_encoding.shape)
