# train_utils.py

import torch
from torch.amp import autocast, GradScaler
from model_evidential import evidential_loss

def train_epoch(model, loader, optimizer, device, lambda_coef, risk_params):
    """
    Esegue un’epoca di training in mixed precision (su CUDA) o full precision (su CPU):
      - forward → genera le evidenze
      - calcola evidential_loss
      - backward + optimizer step con GradScaler
    Ritorna la loss media dell’epoca.
    """
    model.train()
    scaler = GradScaler()

    running_loss = 0.0
    total_samples = 0

    # Determina se abilitare autocast (solo su CUDA)
    use_amp = (device.type == "cuda")

    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device, non_blocking=True), Yb.to(device, non_blocking=True)
        optimizer.zero_grad()

        # Passiamo device_type e enabled ad autocast
        with autocast(device_type=device.type, enabled=use_amp):
            ev = model(Xb)  # (batch, num_classes)
            loss = evidential_loss(
                Yb, ev,
                lambda_coef        = lambda_coef,
                risk_weight        = risk_params["risk_weight"],
                target_uncertainty = risk_params["target_uncertainty"],
                num_classes        = Yb.size(1)
            )

        # Mixed‐precision backward + step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accumula per la loss media
        bsize = Xb.size(0)
        running_loss += loss.item() * bsize
        total_samples += bsize

    return (running_loss / total_samples) if total_samples else 0.0
