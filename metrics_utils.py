import torch

def compute_metrics(model, loader, device, num_classes):
    model.eval()
    corr=tot=0
    sce= sme= scu= smu= 0.0
    cntc= cntm = 0
    with torch.no_grad():
        for Xb,Yb in loader:
            Xb,Yb = Xb.to(device), Yb.to(device)
            ev    = model(Xb)
            alpha = ev + 1.0
            S     = alpha.sum(dim=1, keepdim=True)
            p     = alpha/S
            preds = p.argmax(dim=1)
            labels= Yb.argmax(dim=1)
            mask  = preds.eq(labels)

            corr += mask.sum().item()
            tot  += Xb.size(0)

            te = alpha.sum(dim=1)
            un = num_classes/(S.squeeze(1)+num_classes)

            sce += te[mask].sum().item()
            sme += te[~mask].sum().item()
            scu += un[mask].sum().item()
            smu += un[~mask].sum().item()
            cntc+= mask.sum().item()
            cntm+= (~mask).sum().item()

    acc     = corr/tot if tot else 0.0
    avg_ce  = sce/cntc if cntc else 0.0
    avg_me  = sme/cntm if cntm else 0.0
    avg_cu  = scu/cntc if cntc else 0.0
    avg_mu  = smu/cntm if cntm else 0.0

    return acc, avg_ce, avg_me, avg_cu, avg_mu
