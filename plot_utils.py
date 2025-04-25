import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_curves(tr, te, out="article_style_evidence_uncertainty.png"):
    E = np.arange(1, len(tr["accuracy"])+1)
    fig, ax = plt.subplots(2,2,figsize=(12,10))
    # (a) Training evidence
    a0 = ax[0,0]
    a0.plot(E, tr["corr_evd"], 'r-', label="correct")
    a0.plot(E, tr["mis_evd"],  'k-', label="misclass")
    a0.set(title="(a) Training evidence", xlabel="Epoch", ylabel="Total evidence")
    a0.legend()

    # (b) Training acc & unc
    a1 = ax[0,1]
    a1.plot(E, tr["accuracy"], 'b-',  label="accuracy")
    a1.plot(E, tr["corr_unc"], 'r--', label="unc corr")
    a1.plot(E, tr["mis_unc"],  'k--', label="unc mis")
    a1.set(title="(b) Training acc & unc", xlabel="Epoch", ylabel="Value")
    a1.legend()

    # (c) Testing evidence
    a2 = ax[1,0]
    a2.plot(E, te["corr_evd"], 'r-')
    a2.plot(E, te["mis_evd"],  'k-')
    a2.set(title="(c) Testing evidence", xlabel="Epoch", ylabel="Total evidence")
    a2.legend(["correct","misclass"])

    # (d) Testing **uncertainty** only
    a3 = ax[1,1]
    a3.plot(E, te["corr_unc"], 'r--', label="unc corr")
    a3.plot(E, te["mis_unc"],  'k--', label="unc mis")
    a3.set(title="(d) Testing uncertainty", xlabel="Epoch", ylabel="Uncertainty")
    a3.legend()

    plt.tight_layout()
    plt.savefig(out)
    plt.show()

def plot_sample_bar(model, X_full, device, num_classes, out="bar_10_samples.png"):
    model.eval()
    idx = torch.randperm(len(X_full))[:10]
    X10 = X_full[idx].to(device)
    with torch.no_grad():
        ev = model(X10)
        alpha = ev+1
        S     = alpha.sum(1,keepdim=True)
        P     = (alpha/S).cpu().numpy()
        U     = (num_classes/(S+num_classes)).squeeze(1).cpu().numpy()

    x = np.arange(10)
    w = 0.08
    fig, ax = plt.subplots(figsize=(10,5))
    for c in range(num_classes):
        ax.bar(x + (c - num_classes/2)*w, P[:,c], w, alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(x, U, 'ko-', label="uncertainty")

    ax.set(xticks=x, xticklabels=[f"s{i+1}" for i in x], xlabel="Sample", ylabel="Probability")
    ax2.set_ylabel("Uncertainty")
    ax.set_title("Class probabilities + uncertainty (10 samples)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out)
    plt.show()
