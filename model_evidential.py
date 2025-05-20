import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialVGG(nn.Module):
    """
    EVGG come da articolo:
    3 blocchi [Conv-Conv-Pool] con canali 32→32→POOL,
    32→64→POOL, 64→128→POOL; poi FC(128*3*3→256) + Dropout + FC(256→C)
    e softplus per l'evidence.
    """
    def __init__(self, num_classes: int, input_channels: int = 1):
        super().__init__()
        # --- Block 1 ---
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        # --- Block 2 ---
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        # --- Block 3 ---
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        # --- Head MLP ---
        self.fc1 = nn.Linear(128*3*3, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        # Xavier init leggermente ridotta
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # reshape invece di view per sicurezza
        x = x.reshape(x.size(0), -1)           
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        raw_e = self.fc2(x)
        # softplus per garantire ≥0
        evidence = F.softplus(raw_e)
        return evidence

def evidential_loss(
    y: torch.Tensor,
    evidence: torch.Tensor,
    lambda_coef: float,
    risk_weight: float,
    target_uncertainty: float,
    num_classes: int
) -> torch.Tensor:
    """
    1) Data term: err^2 + var
    2) KL divergence Dir(alpha)||Dir(1)
    3) Risk sul valore di incertezza
    """
    alpha = evidence + 1.0
    S     = alpha.sum(dim=1, keepdim=True)

    # 1) data term = errore^2 + varianza
    p   = alpha / S
    err = ((y - p) ** 2).sum(dim=1)
    var = (p * (1 - p) / S).sum(dim=1)
    data_term = (err + var).mean()

    # 2) KL div. Dir(alpha)||Dir(1)
    kl = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1)
        + ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(S))).sum(dim=1)
    )
    kl_term = kl.mean()

    # 3) risk term sull’incertezza
    uncertainty = num_classes / (S + num_classes)
    risk_term  = ((uncertainty - target_uncertainty) ** 2).mean()

    return data_term + lambda_coef * kl_term + risk_weight * risk_term