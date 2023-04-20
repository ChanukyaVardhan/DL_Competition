from x3d import *

# model = X3D(width_factor = 1.0, depth_factor = 1.0, bottleneck_factor = 1.0, num_classes = 49)
model = X3D(width_factor = 1.0, depth_factor = 2.2, bottleneck_factor = 2.25, num_classes = 400)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Total number of trainable parameters: {total_params}')

x = torch.randn(1, 10, 3, 11, 224, 224)
# 1 -> Path, Should be 1 always
# 10 -> Batch Size
# 3 -> Num Channels
# 11 -> Num Frames
# 224 -> H
# 224 -> W

model(x).shape

child = list(model.children())
print(child)
