import torch
import torch.nn as nn
import torch.nn.functional as F

class EvidentialVGG(nn.Module):
    """
    Versione “deeper” in stile VGG per far crescere meglio l’evidence.
    3 blocchi conv+conv+pool, poi FC.
    """
    def __init__(self, num_classes, input_channels=1):
        super(EvidentialVGG, self).__init__()
        # --- Block 1 ---
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64,             64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        # --- Block 2 ---
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128,128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        # --- Block 3 ---
        self.conv5 = nn.Conv2d(128,256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        # --- Fully connected ---
        self.fc1   = nn.Linear(256*3*3, 512)  # 28→14→7→3
        self.drop  = nn.Dropout(0.4)
        self.fc2   = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x)); x = self.pool1(x)
        x = F.relu(self.conv3(x)); x = F.relu(self.conv4(x)); x = self.pool2(x)
        x = F.relu(self.conv5(x)); x = F.relu(self.conv6(x)); x = self.pool3(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        evidence = F.softplus(self.fc2(x))
        return evidence


def evidential_loss(y, evidence, lambda_coef, risk_weight, target_uncertainty, num_classes):
    """
    DataTerm (err+var) + KL vs Dir(1) + Risk on uncertainty
    """
    alpha = evidence + 1.0
    S     = alpha.sum(dim=1, keepdim=True)

    # 1) data term
    p   = alpha / S
    err = ((y - p)**2).sum(dim=1)
    var = (p*(1-p)/S).sum(dim=1)
    data_term = (err + var).mean()

    # 2) KL divergence
    kl = (
        torch.lgamma(S)
        - torch.lgamma(alpha).sum(dim=1)
        + ((alpha-1)*(torch.digamma(alpha)-torch.digamma(S))).sum(dim=1)
    )
    kl_term = kl.mean()

    # 3) risk term
    unc = num_classes / (S + num_classes)
    risk_term = ((unc - target_uncertainty)**2).mean()

    return data_term + lambda_coef*kl_term + risk_weight*risk_term
