import torch
from torch.amp import autocast, GradScaler
from model_evidential import evidential_loss

def train_epoch(model, loader, optimizer, device, lambda_coef, risk_params):
    """
    Una epoca di training in mixed precision (se GPU disponibile).
    """
    model.train()
    scaler = GradScaler(device_type="cuda" if torch.cuda.is_available() else "cpu")

    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device, non_blocking=True), Yb.to(device, non_blocking=True)
        optimizer.zero_grad()

        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            ev   = model(Xb)
            loss = evidential_loss(
                Yb, ev,
                lambda_coef=lambda_coef,
                risk_weight=risk_params["risk_weight"],
                target_uncertainty=risk_params["target_uncertainty"],
                num_classes=Yb.size(1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
