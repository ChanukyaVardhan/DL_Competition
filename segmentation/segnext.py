import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, cardinality, width, reduction):
        super(SEResNeXtBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels * width,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels * width)
        self.conv2 = nn.Conv2d(out_channels * width, out_channels * width,
                               kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * width)
        self.conv3 = nn.Conv2d(out_channels * width, out_channels *
                               self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.se_layer = SELayer(out_channels * self.expansion, reduction)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se_layer(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out


class ModifiedSegNeXT(nn.Module):
    def __init__(self, num_classes, embedding_dim, feature_map_channels, height, width, pretrained=False):
        super(ModifiedSegNeXT, self).__init__()

        # Projector to convert to 2d dimensions.
        self.fc = nn.Linear(
            embedding_dim, feature_map_channels * height * width)

        # Add SEResNeXt blocks for decoder
        self.decoder = nn.Sequential(
            SEResNeXtBottleneck(2048, 512, 1, 32, 4, 16),
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=0),
            SEResNeXtBottleneck(1024, 256, 1, 32, 4, 16),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0),
            SEResNeXtBottleneck(512, 128, 1, 32, 4, 16),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
            SEResNeXtBottleneck(256, 64, 1, 32, 4, 16),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 2048, 7, 7)
        x = self.decoder(x)
        return x


class SegNeXT(nn.Module):
    def __init__(self, num_classes, weights=None):
        super(SegNeXT, self).__init__()

        # Load pre-trained ResNeXt50-32x4d model
        base_model = resnext50_32x4d(weights=weights)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])

        # Replace first convolution layer to accept different number of input channels
        self.encoder[0] = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Add SEResNeXt blocks for decoder
        self.decoder = nn.Sequential(
            SEResNeXtBottleneck(2048, 512, 1, 32, 4, 16),
            nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2, padding=0),
            SEResNeXtBottleneck(1024, 256, 1, 32, 4, 16),
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0),
            SEResNeXtBottleneck(512, 128, 1, 32, 4, 16),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0),
            SEResNeXtBottleneck(256, 64, 1, 32, 4, 16),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = ModifiedSegNeXT(10, 512, 256, 8, 8)
    print(count_parameters(model))
