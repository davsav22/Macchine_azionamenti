# train_utils.py

import torch
from torch.amp import autocast, GradScaler
from model_evidential import evidential_loss

def train_epoch(model, loader, optimizer, device,
                lambda_coef, risk_params):
    use_amp = (device.type == 'cuda')
    scaler  = GradScaler(enabled=use_amp)

    model.train()
    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device, non_blocking=True), Yb.to(device, non_blocking=True)
        optimizer.zero_grad()

        if use_amp:
            with autocast('cuda', enabled=True):
                ev   = model(Xb)
                loss = evidential_loss(
                    Yb, ev,
                    lambda_coef        = lambda_coef,
                    risk_weight        = risk_params["risk_weight"],
                    target_uncertainty = risk_params["target_uncertainty"],
                    num_classes        = Yb.size(1)
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            ev   = model(Xb)
            loss = evidential_loss(
                Yb, ev,
                lambda_coef        = lambda_coef,
                risk_weight        = risk_params["risk_weight"],
                target_uncertainty = risk_params["target_uncertainty"],
                num_classes        = Yb.size(1)
            )
            loss.backward()
            optimizer.step()
