# model_evidential.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialVGG(nn.Module):
    """
    “Evidential VGG”: 3 blocchi Conv-Conv-Pool (28→14→7→3), poi MLP per l'evidence.
    """
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        # --- Block 1 (28→14) ---
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # --- Block 2 (14→7) ---
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # --- Block 3 (7→3) ---
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        # --- Head MLP ---
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # ora x ha shape [B,128,3,3]
        x = x.reshape(x.size(0), -1)       # .reshape() garantisce continuità
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        evidence = F.softplus(self.fc2(x)) # >0
        return evidence


def evidential_loss(y, evidence,
                   lambda_coef:float,
                   risk_weight:float,
                   target_uncertainty:float,
                   num_classes:int):
    alpha = evidence + 1.0
    S     = alpha.sum(dim=1, keepdim=True)
    p     = alpha/S

    # 1) data term
    err = ((y-p)**2).sum(dim=1)
    var = (p*(1-p)/S).sum(dim=1)
    data_term = (err+var).mean()

    # 2) KL Dir(alpha)||Dir(1)
    kl = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(1)
        + ((alpha-1)*(torch.digamma(alpha)-torch.digamma(S))).sum(1)
    )
    kl_term = kl.mean()

    # 3) risk term
    unc = num_classes/(S+num_classes)
    risk_term = ((unc-target_uncertainty)**2).mean()

    return data_term + lambda_coef*kl_term + risk_weight*risk_term
