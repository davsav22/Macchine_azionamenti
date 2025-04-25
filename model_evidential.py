import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialVGG(nn.Module):
    """
    Versione VGG-like più profonda:
    ConvBlock x4 +  FC + Dropout + FC(evidence)
    """
    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2,2)
            )
        self.features = nn.Sequential(
            conv_block(in_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 256),   # 28→14→7→3→1
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        # softplus per avere evidence>=0
        return F.softplus(x)
