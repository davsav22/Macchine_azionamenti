# train_utils.py

import torch
from torch.amp import autocast, GradScaler
from model_evidential import evidential_loss

def train_epoch(model, loader, optimizer, device,
                lambda_coef: float, risk_params: dict):
    """
    Esegue un'epoca di training:
      - Se CUDA è disponibile, usa AMP (autocast + GradScaler).
      - Altrimenti, fa il training in float32 standard.
    """
    use_amp = (device.type == "cuda")
    scaler  = GradScaler(enabled=use_amp)

    model.train()
    for batch_idx, (Xb, Yb) in enumerate(loader):
        Xb, Yb = Xb.to(device, non_blocking=True), Yb.to(device, non_blocking=True)
        optimizer.zero_grad()

        # ---- forward + loss ----
        if use_amp:
            # autocast accetta solo 'cuda' o 'cpu' come device_type
            with autocast(device_type="cuda", enabled=True):
                ev   = model(Xb)
                loss = evidential_loss(
                    Yb, ev,
                    lambda_coef        = lambda_coef,
                    risk_weight        = risk_params["risk_weight"],
                    target_uncertainty = risk_params["target_uncertainty"],
                    num_classes        = Yb.size(1)
                )
            # backward scalato + step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # semplice float32 su CPU
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

        # stampa di debug ogni 10 batch
        if batch_idx % 10 == 0:
            print(f"  [Batch {batch_idx:3d}] loss = {loss.item():.4f}")
