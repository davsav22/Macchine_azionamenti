import torch
from torch.cuda.amp import autocast, GradScaler
from model_evidential import evidential_loss

def train_epoch(model, loader, optimizer, device,
                lambda_coef, risk_params):
    """
    Un’epoca di training in mixed precision (se CUDA disponibile).
    """
    model.train()
    use_amp = (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)

    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad()

        with autocast(enabled=use_amp):
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
