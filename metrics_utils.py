import torch

def compute_metrics(model, loader, device, num_classes):
    model.eval()
    corr=tot=0
    sum_ce=sum_me=sum_cu=sum_mu=0.0
    cnt_c=cnt_m=0

    with torch.no_grad():
        for Xb,Yb in loader:
            Xb,Yb = Xb.to(device), Yb.to(device)
            ev = model(Xb)
            alpha = ev+1
            S     = alpha.sum(1,keepdim=True)
            p     = alpha/S

            preds  = p.argmax(1)
            labels = Yb.argmax(1)
            mask   = preds.eq(labels)

            corr += mask.sum().item()
            tot  += Xb.size(0)

            te = alpha.sum(1)
            un = num_classes/(S.squeeze(1)+num_classes)

            sum_ce += te[mask].sum().item()
            sum_me += te[~mask].sum().item()
            sum_cu += un[mask].sum().item()
            sum_mu += un[~mask].sum().item()
            cnt_c  += mask.sum().item()
            cnt_m  += (~mask).sum().item()

    acc   = corr/tot if tot else 0.0
    ace   = sum_ce/cnt_c if cnt_c else 0.0
    ame   = sum_me/cnt_m if cnt_m else 0.0
    auc   = sum_cu/cnt_c if cnt_c else 0.0
    amu   = sum_mu/cnt_m if cnt_m else 0.0

    return acc, ace, ame, auc, amu
