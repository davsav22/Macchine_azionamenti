import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialVGG(nn.Module):
    """
    Architettura più profonda in stile VGG per far salire l'evidence.
    3×(Conv-Conv-Pool) + FC.
    """
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        # Block 1
        self.conv1 = nn.Conv2d(input_channels,  32, 3, padding=1)
        self.conv2 = nn.Conv2d(32,             32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        # Block 2
        self.conv3 = nn.Conv2d(32,             64, 3, padding=1)
        self.conv4 = nn.Conv2d(64,             64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        # Block 3
        self.conv5 = nn.Conv2d(64,            128, 3, padding=1)
        self.conv6 = nn.Conv2d(128,           128, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        # FC
        self.fc1   = nn.Linear(128*3*3, 256)  # 28→14→7→3
        self.drop  = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = self.pool1(x)
        x = F.relu(self.conv3(x)); x = F.relu(self.conv4(x)); x = self.pool2(x)
        x = F.relu(self.conv5(x)); x = F.relu(self.conv6(x)); x = self.pool3(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        # softplus garantisce evidence ≥0
        evidence = F.softplus(self.fc2(x))
        return evidence

def evidential_loss(y, evidence,
                   lambda_coef: float,
                   risk_weight: float,
                   target_uncertainty: float,
                   num_classes: int):
    alpha = evidence + 1.0
    S     = alpha.sum(dim=1, keepdim=True)

    # 1) Data term: errore quadratico + varianza
    p   = alpha / S
    err = ((y - p)**2).sum(dim=1)
    var = (p*(1-p)/S).sum(dim=1)
    data_term = (err + var).mean()

    # 2) KL(D(alpha) || D(1))
    kl = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1)
        + ((alpha-1)*(torch.digamma(alpha)-torch.digamma(S))).sum(dim=1)
    )
    kl_term = kl.mean()

    # 3) Risk term sull’incertezza
    unc = num_classes / (S + num_classes)
    risk_term = ((unc - target_uncertainty)**2).mean()

    return data_term + lambda_coef*kl_term + risk_weight*risk_term
