import torch
import torch.nn as nn
from .ARConv import ARConv # Make sure ARConv.py is in the same folder

class BasicConv(nn.Module):
    """
    A basic convolutional block for the network entrance.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AR_ResBlock(nn.Module):
    """
    Residual Block using ARConv.
    """
    def __init__(self, in_channels, out_channels):
        super(AR_ResBlock, self).__init__()
        # If input and output channels are different, use a 1x1 convolution for the shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.shortcut = None

        # Adapter convolution to match channels for ARConv and the main path
        self.conv_adapter = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ARConv layers
        self.ar_conv1 = ARConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.ar_conv2 = ARConv(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, epoch, hw_range):
        # The main path starts with the adapter
        out = self.conv_adapter(x)
        
        # Prepare the identity connection (shortcut)
        if self.shortcut:
            identity = self.shortcut(x)
        else:
            identity = out # The identity is taken after channel adaptation

        # First ARConv layer
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ar_conv1(out, epoch, hw_range)

        # Second ARConv layer
        out = self.bn2(out)
        out = self.relu(out)
        out = self.ar_conv2(out, epoch, hw_range)

        # Add the shortcut
        out += identity
        return out

class MSHNet_AR(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(MSHNet_AR, self).__init__()

        self.conv_in = BasicConv(in_channels, 64)

        # Encoder Path
        self.encoder1 = AR_ResBlock(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = AR_ResBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = AR_ResBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Middle Block
        self.middle_block = AR_ResBlock(256, 512)

        # Decoder Path
        self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = AR_ResBlock(512, 256) # Input channels: 256 (from upsample) + 256 (from x3) = 512
        
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = AR_ResBlock(256, 128) # Input channels: 128 (from upsample) + 128 (from x2) = 256
        
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = AR_ResBlock(128, 64) # Input channels: 64 (from upsample) + 64 (from x1) = 128

        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def forward(self, x, epoch, hw_range=[1, 9]):
        x_in = self.conv_in(x)

        # Encoder
        x1 = self.encoder1(x_in, epoch, hw_range)
        x_pool1 = self.pool1(x1)
        x2 = self.encoder2(x_pool1, epoch, hw_range)
        x_pool2 = self.pool2(x2)
        x3 = self.encoder3(x_pool2, epoch, hw_range)
        x_pool3 = self.pool3(x3)

        # Middle
        x_middle = self.middle_block(x_pool3, epoch, hw_range)

        # Decoder
        x_up3 = self.upsample3(x_middle)
        x_cat3 = torch.cat((x_up3, x3), dim=1)
        x_de3 = self.decoder3(x_cat3, epoch, hw_range)

        x_up2 = self.upsample2(x_de3)
        x_cat2 = torch.cat((x_up2, x2), dim=1)
        x_de2 = self.decoder2(x_cat2, epoch, hw_range)

        x_up1 = self.upsample1(x_de2)
        x_cat1 = torch.cat((x_up1, x1), dim=1)
        x_de1 = self.decoder1(x_cat1, epoch, hw_range)

        out = self.conv_out(x_de1)
        
        return out
