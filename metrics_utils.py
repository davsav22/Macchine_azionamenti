import torch

def compute_metrics(model, loader, device, num_classes):
    model.eval()
    corr = tot = 0
    sum_ce = sum_me = sum_cu = sum_mu = 0.0
    cnt_c = cnt_m = 0

    with torch.no_grad():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            ev = model(X)           # (B,C)
            alpha = ev + 1.0
            S     = alpha.sum(1,keepdim=True)
            p     = alpha / S
            preds = p.argmax(1)
            labs  = Y.argmax(1)
            mask  = preds.eq(labs)
            corr += mask.sum().item()
            tot  += X.size(0)

            te = alpha.sum(1)                 # total evidence
            un = (num_classes / (S + num_classes)).squeeze(1)

            sum_ce += te[mask].sum().item()
            sum_me += te[~mask].sum().item()
            sum_cu += un[mask].sum().item()
            sum_mu += un[~mask].sum().item()

            cnt_c += mask.sum().item()
            cnt_m += (~mask).sum().item()

    acc    = corr/tot if tot else 0.0
    avg_ce = sum_ce/cnt_c if cnt_c else 0.0
    avg_me = sum_me/cnt_m if cnt_m else 0.0
    avg_cu = sum_cu/cnt_c if cnt_c else 0.0
    avg_mu = sum_mu/cnt_m if cnt_m else 0.0

    return acc, avg_ce, avg_me, avg_cu, avg_mu
