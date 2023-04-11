import resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer

class PreTrainModel(nn.Module):

	def __init__(self, cnn_encoder = "resnet18", d_emb = 512, d_ff = 2048, n_heads = 8, n_layers = 6, dropout = 0.):
		super(PreTrainModel, self).__init__()

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

	def forward(self, input_images, input_frames, pred_image, pred_frame):
		B, Nf, C, H, W 		= input_images.shape
		# input_images: B x Nf x 3 x 224 x 224 -> input_images_: B*Nf x 3 x 224 x 224
		input_images_ 		= input_images.reshape(B*Nf, C, H, W)
		# input_images_: B*Nf x 3 x 224 x 224 -> input_images_enc: B x Nf x d_emb
		input_images_enc 	= self.cnn_encoder(input_images_)
		input_images_enc 	= input_images_enc.reshape(B, Nf, input_images_enc.shape[-1])

		# pred_image_enc: B x 1 x d_emb
		pred_image_enc 		= nn.Parameter(torch.randn(B, 1, self.d_emb))

		x 					= self.positional_encoding(
			torch.cat([input_images_enc, pred_image_enc], dim = 1),
			torch.cat([input_frames, pred_frame], dim = 1)
		)

		# x_encoding: B x (Nf + 1) x d_emb
		x_encoding 			= self.transformer_encoder(x, mask = None)
		x_encoding 			= self.mlp(x_encoding)

		# pred_image: B x 3 x 224 x 224 -> y_encoding: B x d_emb
		y_encoding 			= self.cnn_encoder(pred_image)

		return x_encoding, y_encoding

class VICReg(nn.Module):

	def __init__(self, sim_coeff = 25, std_coeff = 25, cov_coeff = 1):
		super(VICReg, self).__init__()

		self.sim_coeff = sim_coeff
		self.std_coeff = std_coeff
		self.cov_coeff = cov_coeff

	def forward(self, x_encoding, y_encoding):
		# x_encoding: B x (Nf + 1) x d_emb, y_encoding: B x d_emb

		repr_loss = F.mse_loss(x_encoding[:, -1, :], y_encoding)

		# FIX THIS - HOW TO INCORPORATE THE STD_LOSS AND COV_LOSS HERE?

		# loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss
		loss = self.sim_coeff * repr_loss

		return loss
