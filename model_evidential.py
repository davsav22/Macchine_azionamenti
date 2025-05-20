import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialVGG(nn.Module):
    """
    “Evidential VGG”: 3 blocchi Conv-Conv-Pool, poi MLP per evidenza.
    """
    def __init__(self, num_classes: int, input_channels: int = 1):
        super().__init__()
        # Blocchi convoluzionali
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
        )
        # Head MLP
        self.fc1 = nn.Linear(128*3*3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.reshape(x.size(0), -1)        # ← usa reshape, non view
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        evidence = F.softplus(self.fc2(x))
        return evidence

def evidential_loss(
    y: torch.Tensor,
    evidence: torch.Tensor,
    lambda_coef: float,
    risk_weight: float,
    target_uncertainty: float,
    num_classes: int
) -> torch.Tensor:
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)

    # 1) data term = errore^2 + varianza
    p = alpha / S
    err = ((y - p) ** 2).sum(dim=1)
    var = (p * (1 - p) / S).sum(dim=1)
    data_term = (err + var).mean()

    # 2) KL divergence Dir(alpha) || Dir(1)
    kl = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1)
        + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    )
    kl_term = kl.mean()

    # 3) risk term sull'incertezza
    uncertainty = num_classes / (S + num_classes)
    risk_term = ((uncertainty - target_uncertainty) ** 2).mean()

    return data_term + lambda_coef * kl_term + risk_weight * risk_term
