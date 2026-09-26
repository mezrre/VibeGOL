import torch.nn as nn
import torch.nn.functional as F

# Convulation block used by model. Uses 2 conv layers with batch norm
class ConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=0)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=0)
        self.bn2 = nn.BatchNorm2d(channels)

    def circ_pad(self, x):
        return F.pad(x, (1, 1, 1, 1), mode="circular")

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(self.circ_pad(x))))
        out = self.bn2(self.conv2(self.circ_pad(out)))
        return F.relu(out + residual)

# VibeGOL model definition
# Simple architecture: Spam Convulutional layers and relu
# If the mode is not performing, try adding more layers in config/training_config.json
# Model doing bad? more layers. model guessing wrong? more layers. Life making you sad? more layers.
class GoLReverseNet(nn.Module):
    def __init__(self, channels=64, n_blocks=24):
        super().__init__()

        # Three input channels:
        # next state, masked predecessor, predecessor mask.
        self.stem = nn.Conv2d(3, channels, 3, padding=0)
        self.stem_pad = lambda x: F.pad(x, (1, 1, 1, 1), mode="circular")
        self.blocks = nn.ModuleList([ConvBlock(channels) for _ in range(n_blocks)])
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        x = F.relu(self.stem(self.stem_pad(x)))
        for block in self.blocks:
            x = block(x)
        return self.head(x)