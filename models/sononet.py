import torch
import torch.nn as nn
import torch.nn.functional as F

class SonoNet16(nn.Module):
    """
    Simplified SonoNet-16 architecture for 2D slice classification.
    Designed for single-channel ADNI slices and multi-class classification (3 classes).
    """
    def __init__(self, in_channels=1, num_classes=3):
        super(SonoNet16, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)

        self.pool = nn.MaxPool2d(2, 2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.enc3(x)
        x = self.pool(x)
        x = self.enc4(x)
        x = self.pool(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
