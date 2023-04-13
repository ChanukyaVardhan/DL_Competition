import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer

class PreTrainModel(nn.Module):

	def __init__(self, cnn_encoder = "resnet18", d_emb = 512, d_ff = 2048, n_heads = 8, n_layers = 6, dropout = 0.):
		super(PreTrainModel, self).__init__()

		# FIX - CHECK IF WE WANT TO USE RESNET FROM PYTORCH DIRECTLY
		self.cnn_encoder 			= self._get_cnn_encoder(cnn_encoder)
		self.d_emb 					= d_emb
		self.d_ff 					= d_ff
		self.n_heads 				= n_heads
		self.n_layers 				= n_layers
		self.dropout 				= dropout

		self.transformer_encoder 	= transformer.TransformerEncoder(self.d_emb, self.d_ff, self.n_heads, self.n_layers, self.dropout)
		self.positional_encoding 	= transformer.PositionalEncoding(self.d_emb, self.dropout)

		# DO WE NEED A NON LINEAR ACTIVATION HERE?
		self.mlp 					= nn.Linear(self.d_emb, self.d_emb)

	def _get_cnn_encoder(self, cnn_encoder):
		if cnn_encoder == "resnet18":
			return resnet.resnet18()
		elif cnn_encoder == "resnet34":
			return resnet.resnet34()
		elif cnn_encoder == "resnet50":
			return resnet.resnet50()
		elif cnn_encoder == "resnet101":
			return resnet.resnet101()
		elif cnn_encoder == "resnet152":
			return resnet.resnet152()
		else:
			raise Exception(f'{cnn_encoder} is not a valid CNN encoder!')

	def forward(self, input_images, input_frames, start_frame, pred_image, pred_frame):
		B, Nf, C, H, W 		= input_images.shape
		# input_images: B x Nf x 3 x 224 x 224 -> input_images_: B*Nf x 3 x 224 x 224
		input_images_ 		= input_images.reshape(B*Nf, C, H, W)
		# input_images_: B*Nf x 3 x 224 x 224 -> x_encoding: B x Nf x d_emb
		input_images_enc 	= self.cnn_encoder(input_images_)
		x_encoding 			= input_images_enc.reshape(B, Nf, -1)

		# pred_image_enc: B x 1 x d_emb
		pred_image_enc 		= nn.Parameter(torch.randn(B, 1, self.d_emb))

		x 					= self.positional_encoding(
			torch.cat([x_encoding, pred_image_enc], dim = 1),
			torch.cat([input_frames - start_frame, pred_frame - start_frame], dim = 1) # Subtracting start_frame so transformer sees this without offset
		)

		# x_encoding_pred: B x d_emb
		x_encoding_pred 	= self.transformer_encoder(x, mask = None)
		x_encoding_pred 	= self.mlp(x_encoding_pred[:, -1, :])

		# pred_image: B x 3 x 224 x 224 -> y_encoding: B x d_emb
		y_encoding 			= self.cnn_encoder(pred_image)

		return x_encoding, x_encoding_pred, y_encoding

class VICReg(nn.Module):

	def __init__(self, sim_coeff = 25, std_coeff = 25, cov_coeff = 1):
		super(VICReg, self).__init__()

		self.sim_coeff = sim_coeff
		self.std_coeff = std_coeff
		self.cov_coeff = cov_coeff

	def off_diagonal(self, x):
		if x.dim() == 3:
			_, n, m = x.shape
			assert n == m
			return x.flatten(1)[:, :-1].view(-1, n - 1, n + 1)[:, :, 1:].flatten(1)
		else:
			n, m = x.shape
			assert n == m
			return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

	def forward(self, x_encoding, x_encoding_pred, y_encoding):
		# x_encoding: B x Nf x d_emb, x_encoding_pred: B x d_emb, y_encoding: B x d_emb
		B, _, d_emb = x_encoding.shape

		repr_loss = F.mse_loss(x_encoding_pred, y_encoding)

		x_encoding = x_encoding - x_encoding.mean(dim=0)
		y_encoding = y_encoding - y_encoding.mean(dim=0)

		std_x = torch.sqrt(x_encoding.var(dim=0) + 0.0001)
		std_y = torch.sqrt(y_encoding.var(dim=0) + 0.0001)
		std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

		cov_x = (x_encoding.permute(1, 2, 0) @ x_encoding.permute(1, 0, 2)) / (B - 1)
		cov_y = (y_encoding.T @ y_encoding) / (B - 1)
        # FIX - SHOULD THIS BE A MEAN OR SUM(), WEIGHT FOR COV_LOSS DEPENDS ON THAT
		cov_loss = self.off_diagonal(cov_x).pow_(2).sum().div(d_emb).mean() + \
        			self.off_diagonal(cov_y).pow_(2).sum().div(d_emb)

		loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

		return loss
