# model_evidential.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialCNN(nn.Module):
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        self.conv1   = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(128, num_classes)

        # *** inizializziamo bias e pesi finali ***
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # raw output
        raw_e = self.fc2(x)
        # evidence zero-centered: softplus-1, clamp per sicurezza
        evidence = F.softplus(raw_e) - 1.0
        return torch.clamp(evidence, min=0.0)
