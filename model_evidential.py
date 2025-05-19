# model_evidential.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialVGG(nn.Module):
    """
    EVGG: 3 blocchi Conv-Conv-Pool (32→64→128 feature map), poi MLP con head evidenziale.
    Replicazione fedele della struttura usata nell’articolo.
    """

    def __init__(self, num_classes: int, input_channels: int = 1):
        super().__init__()
        # --- Block 1: conv3x3(32) → conv3x3(32) → pool ---
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # --- Block 2: conv3x3(64) → conv3x3(64) → pool ---
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # --- Block 3: conv3x3(128) → conv3x3(128) → pool ---
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # --- Head MLP (da 28×28 input, con 3 pool riduce a 28→14→7→3) ---
        # OUTPUT PRODUCE “evidence” per ciascuna classe
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # applica i tre blocchi
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # flatten
        x = x.view(x.size(0), -1)
        # MLP
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # head evidenziale: softplus garantisce ev ≥ 0
        evidence = F.softplus(self.fc2(x))
        return evidence
