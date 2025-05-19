# train_utils.py

import torch
from torch.cuda.amp import autocast, GradScaler
from model_evidential import evidential_loss

def train_epoch(model, loader, optimizer, device, lambda_coef, risk_params):
    """
    Un'epoca di training in mixed precision su GPU (o fallback su CPU):
      - se CUDA è presente, entra in autocast()
      - altrimenti esegue in FP32
      - usa GradScaler senza argomenti per evitare deprecations
    """
    model.train()
    scaler = GradScaler()  # su CPU è no-op; su GPU usa AMP

    for Xb, Yb in loader:
        Xb, Yb = Xb.to(device, non_blocking=True), Yb.to(device, non_blocking=True)
        optimizer.zero_grad()

        if device.type == 'cuda':
            # Mixed-precision sul CUDA
            with autocast():
                ev   = model(Xb)
                loss = evidential_loss(
                    Yb, ev,
                    lambda_coef=lambda_coef,
                    risk_weight       = risk_params["risk_weight"],
                    target_uncertainty= risk_params["target_uncertainty"],
                    num_classes       = Yb.size(1)
                )
        else:
            # FP32 puro su CPU
            ev   = model(Xb)
            loss = evidential_loss(
                Yb, ev,
                lambda_coef=lambda_coef,
                risk_weight       = risk_params["risk_weight"],
                target_uncertainty= risk_params["target_uncertainty"],
                num_classes       = Yb.size(1)
            )

        # backward + step con AMP scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
