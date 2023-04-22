from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

import math
import resnet_helper
import torch
import torch.nn as nn
import torch.nn.functional as F

def round_width(width, multiplier, min_width=1, divisor=1):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)

class X3DStem(nn.Module):
    """
    X3D's 3D stem module.
    Performs a spatial followed by a depthwise temporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
        """
        super(X3DStem, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module)

    def _construct_stem(self, dim_in, dim_out, norm_module):
        self.conv_xy = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=(1, self.kernel[1], self.kernel[2]),
            stride=(1, self.stride[1], self.stride[2]),
            padding=(0, self.padding[1], self.padding[2]),
            bias=False,
        )
        self.conv = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=(self.kernel[0], 1, 1),
            stride=(self.stride[0], 1, 1),
            padding=(self.padding[0], 0, 0),
            bias=False,
            groups=dim_out,
        )

        self.bn = norm_module(
            num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
        )
        self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        x = self.conv_xy(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

def get_stem_func(name):
    """
    Retrieves the stem module by name.
    """
    trans_funcs = {"x3d_stem": X3DStem}
    assert (
        name in trans_funcs.keys()
    ), "Transformation function '{}' not supported".format(name)
    return trans_funcs[name]

class VideoModelStem(nn.Module):
    """
    Video 3D stem module. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        stem_func_name="basic_stem",
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, Slow
        and etc), list size of 2 for two pathway models (SlowFast).
        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            stem_func_name (string): name of the the stem function applied on
                input to the network.
        """
        super(VideoModelStem, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            )
            == 1
        ), "Input pathway dimensions are not consistent. {} {} {} {} {}".format(
            len(dim_in),
            len(dim_out),
            len(kernel),
            len(stride),
            len(padding),
        )

        self.num_pathways = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.eps = eps
        self.bn_mmt = bn_mmt
        # Construct the stem layer.
        self._construct_stem(dim_in, dim_out, norm_module, stem_func_name)

    def _construct_stem(self, dim_in, dim_out, norm_module, stem_func_name):
        trans_func = get_stem_func(stem_func_name)

        for pathway in range(len(dim_in)):
            stem = trans_func(
                dim_in[pathway],
                dim_out[pathway],
                self.kernel[pathway],
                self.stride[pathway],
                self.padding[pathway],
                self.inplace_relu,
                self.eps,
                self.bn_mmt,
                norm_module,
            )
            self.add_module("pathway{}_stem".format(pathway), stem)

    def forward(self, x):
        assert (
            len(x) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        # use a new list, don't modify in-place the x list, which is bad for activation checkpointing.
        y = []
        for pathway in range(len(x)):
            m = getattr(self, "pathway{}_stem".format(pathway))
            y.append(m(x[pathway]))
        return y

_POOL1 = {
    "x3d": [[1, 1, 1]],
}

_MODEL_STAGE_DEPTH = {18: (2, 2, 2, 2), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}

# Basis of temporal kernel sizes for each of the stage.
_TEMPORAL_KERNEL_BASIS = {
    "x3d": [
        [[5]],  # conv1 temporal kernels.
        [[3]],  # res2 temporal kernels.
        [[3]],  # res3 temporal kernels.
        [[3]],  # res4 temporal kernels.
        [[3]],  # res5 temporal kernels.
    ],
}

class X3DHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.
        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt
        )
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)

        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt
            )
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif self.act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(self.act_func)
            )

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs[0])
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)
        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x

def init_weights(
    model, fc_init_std=0.01, zero_init_final_bn=True, zero_init_final_conv=False
):
    """
    Performs ResNet style weight initialization.
    Args:
        fc_init_std (float): the expected standard deviation for fc layer.
        zero_init_final_bn (bool): if True, zero initialize the final bn for
            every bottleneck.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            # Note that there is no bias due to BN
            if hasattr(m, "final_conv") and zero_init_final_conv:
                m.weight.data.zero_()
            else:
                """
                Follow the initialization method proposed in:
                {He, Kaiming, et al.
                "Delving deep into rectifiers: Surpassing human-level
                performance on imagenet classification."
                arXiv preprint arXiv:1502.01852 (2015)}
                """
                c2_msra_fill(m)

        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
            if (
                hasattr(m, "transform_final_bn")
                and m.transform_final_bn
                and zero_init_final_bn
            ):
                batchnorm_weight = 0.0
            else:
                batchnorm_weight = 1.0
            if m.weight is not None:
                m.weight.data.fill_(batchnorm_weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            if hasattr(m, "xavier_init") and m.xavier_init:
                c2_xavier_fill(m)
            else:
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
            if m.bias is not None:
                m.bias.data.zero_()

class X3D(nn.Module):
    """
    X3D model builder. It builds a X3D network backbone, which is a ResNet.
    Christoph Feichtenhofer.
    "X3D: Expanding Architectures for Efficient Video Recognition."
    https://arxiv.org/abs/2004.04730
    """

    def __init__(self, width_factor = 1.0, depth_factor = 1.0, bottleneck_factor = 1.0, num_classes = 49, cfg = None):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(X3D, self).__init__()

        self.architecture       = "x3d"

        # CHECK - IN CASE WE ARE USING MULTIPLE GPUs, SHOULD THIS BE SYNCBATCHNORM?
        self.norm_module        = nn.BatchNorm3d
        self.enable_detection   = False
        self.num_pathways       = 1

        exp_stage               = 2.0
        # self.dim_c1             = cfg.X3D.DIM_C1
        self.dim_c1             = 12 # Dimensions of the first 3x3 conv layer.

        self.dim_res2           = (
            # round_width(self.dim_c1, exp_stage, divisor=8) if cfg.X3D.SCALE_RES2 else self.dim_c1
            round_width(self.dim_c1, exp_stage, divisor=8) if False else self.dim_c1 # Whether to scale the width of Res2, default is false.
        ) # 24
        self.dim_res3           = round_width(self.dim_res2, exp_stage, divisor=8) # 48
        self.dim_res4           = round_width(self.dim_res3, exp_stage, divisor=8) # 96
        self.dim_res5           = round_width(self.dim_res4, exp_stage, divisor=8) # 192

        self.block_basis = [
            # blocks, channels, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]

        self.resnet_depth = 50 # 18 / 50 / 101

        self.width_factor       = width_factor # Width expansion factor.
        self.depth_factor       = depth_factor # Depth expansion factor.
        self.bottleneck_factor  = bottleneck_factor # Bottleneck expansion factor for the 3x3x3 conv.
        self.num_classes        = num_classes

        self._construct_network(cfg)
        # init_helper.init_weights(
        #     self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        # )
        init_weights(
            self, 0.01, True #  If true, initialize the gamma of the final BN of each block to zero.
        )

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self, cfg):
        """
        Builds a single pathway X3D model.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """

        # assert cfg.MODEL.ARCH in _POOL1.keys()
        assert self.architecture in _POOL1.keys()
        # assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()
        assert self.resnet_depth in _MODEL_STAGE_DEPTH.keys() # Resnet 50 or 18 or 101

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[self.resnet_depth]

        # num_groups      = cfg.RESNET.NUM_GROUPS
        num_groups      = 1 # Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
        # width_per_group = cfg.RESNET.WIDTH_PER_GROUP
        width_per_group = 64 # Width of each group (64 -> ResNet; 4 -> ResNeXt).
        dim_inner       = num_groups * width_per_group

        # w_mul = cfg.X3D.WIDTH_FACTOR
        w_mul = self.width_factor
        # d_mul = cfg.X3D.DEPTH_FACTOR
        d_mul = self.depth_factor
        dim_res1 = round_width(self.dim_c1, w_mul)

        temp_kernel = _TEMPORAL_KERNEL_BASIS[self.architecture]

        # self.s1 = stem_helper.VideoModelStem(
        self.s1 = VideoModelStem(
            # dim_in=cfg.DATA.INPUT_CHANNEL_NUM,
            dim_in=[3],
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name="x3d_stem",
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = round_width(block[1], w_mul)
            # dim_inner = int(cfg.X3D.BOTTLENECK_FACTOR * dim_out)
            dim_inner = int(self.bottleneck_factor * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = "s{}".format(
                stage + 2
            )  # start w res2 to follow convention

            s = resnet_helper.ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                # num_groups=[dim_inner] if cfg.X3D.CHANNELWISE_3x3x3 else [num_groups],
                num_groups=[dim_inner] if True else [num_groups], # Whether to use channelwise (=depthwise) convolution in the center (3x3x3) convolution operation of the residual blocks.
                num_block_temp_kernel=[n_rep],
                # CHECK - ARE THE NONLOCAL VALUES CORRECT?
                # nonlocal_inds=cfg.NONLOCAL.LOCATION[0],
                nonlocal_inds=[[]],
                # nonlocal_group=cfg.NONLOCAL.GROUP[0],
                nonlocal_group=[1],
                # nonlocal_pool=cfg.NONLOCAL.POOL[0],
                nonlocal_pool=[
                        # Res2
                        [[1, 2, 2], [1, 2, 2]],
                        # Res3
                        [[1, 2, 2], [1, 2, 2]],
                        # Res4
                        [[1, 2, 2], [1, 2, 2]],
                        # Res5
                        [[1, 2, 2], [1, 2, 2]],
                    ],
                # instantiation=cfg.NONLOCAL.INSTANTIATION,
                instantiation="dot_product",
                # trans_func_name=cfg.RESNET.TRANS_FUNC,
                trans_func_name="x3d_transform",
                # stride_1x1=cfg.RESNET.STRIDE_1X1,
                stride_1x1=False, # Apply stride to 1x1 conv.
                norm_module=self.norm_module,
                # dilation=cfg.RESNET.SPATIAL_DILATIONS[stage],
                dilation=[1],
                # drop_connect_rate=cfg.MODEL.DROPCONNECT_RATE
                drop_connect_rate=0.5 # CHECK - IS THIS CORRECT?
                * (stage + 2)
                / (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

        if self.enable_detection:
            NotImplementedError
        else:
            # spat_sz = int(math.ceil(cfg.DATA.TRAIN_CROP_SIZE / 32.0))
            spat_sz = int(math.ceil(224 / 32.0)) # CHECK - IS 224 CORRECT?
            # self.head = head_helper.X3DHead(
            self.head = X3DHead(
                dim_in=dim_out,
                dim_inner=dim_inner,
                # dim_out=cfg.X3D.DIM_C5,
                dim_out=2048, # Dimensions of the last linear layer before classificaiton.
                # num_classes=cfg.MODEL.NUM_CLASSES,
                num_classes=self.num_classes, # The number of classes to predict for the model.
                # pool_size=[cfg.DATA.NUM_FRAMES, spat_sz, spat_sz],
        # The number of frames of the input clip.
                pool_size=[11, spat_sz, spat_sz],
                # dropout_rate=cfg.MODEL.DROPOUT_RATE,
                dropout_rate=0.5, # CHECK - IS THIS CORRECT?
                # act_func=cfg.MODEL.HEAD_ACT,
                act_func="softmax",
                # bn_lin5_on=cfg.X3D.BN_LIN5,
                bn_lin5_on=False, # Whether to use a BatchNorm (BN) layer before the classifier, default is false.
            )

    def forward(self, x, bboxes=None):
        for module in self.children():
            x = module(x)
        return x
