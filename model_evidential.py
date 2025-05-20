# model_evidential.py

import torch
import torch.nn as nn
import torch.nn.functional as F

def evidential_loss(y, evidence,
                   lambda_coef: float,
                   risk_weight: float,
                   target_uncertainty: float,
                   num_classes: int):
    """
    Loss evidenziale composita:
      1) Data term = errore (y-p)^2 + varianza p*(1-p)/S
      2) KL[ Dir(alpha) || Dir(1) ] con annealing lambda_coef
      3) Risk term = MSE tra incertezza stimata e target_uncertainty
    """
    # parametri Dirichlet
    alpha = evidence + 1.0              # α = evidence + 1
    S     = alpha.sum(dim=1, keepdim=True)  # somma lungo le classi

    # --- 1) Data term ------------------------------------------------------
    p   = alpha / S                     # expected probabilities
    err = torch.sum((y - p)**2, dim=1)  # squared error per sample
    var = torch.sum(p*(1-p) / S, dim=1) # predictive variance per sample
    data_term = torch.mean(err + var)

    # --- 2) KL divergence vs Dirichlet(1) ---------------------------------
    # KL( Dir(α) || Dir(1) )
    kl = (
        torch.lgamma(S)
        - torch.sum(torch.lgamma(alpha), dim=1)
        + torch.sum((alpha - 1)*(torch.digamma(alpha) - torch.digamma(S)), dim=1)
    )
    kl_term = torch.mean(kl)

    # --- 3) Risk term sull’incertezza -------------------------------------
    # incertezza = M / (S + M)
    unc       = num_classes / (S + num_classes)
    risk_term = torch.mean((unc - target_uncertainty)**2)

    # somma pesata
    return data_term + lambda_coef * kl_term + risk_weight * risk_term


class EvidentialCNN(nn.Module):
    """
    CNN evidenziale ispirata a VGG-lite:
      2 blocchi Conv→ReLU→Pool, poi 2 FC + dropout,
      infine softplus-1 clamp→evidence>=0.
    """
    def __init__(self, num_classes, input_channels=1):
        super().__init__()
        # --- convolutional blocks ---
        self.conv1   = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2   = nn.MaxPool2d(2, 2)

        # --- fully connected layers ---
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2     = nn.Linear(128, num_classes)

        # *** inizializziamo pesi e bias ***
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.constant_(self.fc2.bias, 0.0)

    def forward(self, x):
        # convolutional feature extractor
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # flatten + dense
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # raw logits → evidence = softplus(raw) - 1, clamp≥0
        raw_e    = self.fc2(x)
        evidence = F.softplus(raw_e) - 1.0
        evidence = torch.clamp(evidence, min=0.0)

        return evidence
