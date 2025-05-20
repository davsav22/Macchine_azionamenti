import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_curves(train_hist, test_hist):
    epochs = np.arange(1, len(train_hist["accuracy"]) + 1)
    fig, axes = plt.subplots(2,2, figsize=(12,10))

    # (a) Training evidence
    ax = axes[0,0]
    ax.plot(epochs, train_hist["corr_evd"], 'r-', label="correct")
    ax.plot(epochs, train_hist["mis_evd"],  'k-', label="misclass")
    ax.set_title("(a) Training evidence")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total evidence")
    ax.legend()

    # (b) Training acc & unc
    ax = axes[0,1]
    ax.plot(epochs, train_hist["accuracy"], 'b-',  label="accuracy")
    ax.plot(epochs, train_hist["corr_unc"],  'r--', label="unc corr")
    ax.plot(epochs, train_hist["mis_unc"],   'k--', label="unc mis")
    ax.set_title("(b) Training acc & unc")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value")
    ax.legend()

    # (c) Testing evidence
    ax = axes[1,0]
    ax.plot(epochs, test_hist["corr_evd"], 'r-')
    ax.plot(epochs, test_hist["mis_evd"],  'k-')
    ax.set_title("(c) Testing evidence")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total evidence")
    ax.legend(["correct","misclass"])

    # (d) Testing uncertainty
    ax = axes[1,1]
    ax.plot(epochs, test_hist["corr_unc"], 'r--', label="unc corr")
    ax.plot(epochs, test_hist["mis_unc"],  'k--', label="unc mis")
    ax.set_title("(d) Testing uncertainty")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Uncertainty")
    ax.legend()

    plt.tight_layout()
    plt.savefig("article_style_evidence_uncertainty.png")
    plt.show()

def plot_sample_bar(model, X_full, device, num_classes):
    model.eval()
    idx = torch.randperm(X_full.size(0))[:10]
    X10 = X_full[idx].to(device)
    with torch.no_grad():
        ev    = model(X10)
        alpha = ev + 1
        S     = alpha.sum(1,keepdim=True)
        P     = (alpha/S).cpu().numpy()        # (10,C)
        U     = (num_classes / (S + num_classes)).squeeze(1).cpu().numpy()

    fig, ax = plt.subplots(figsize=(10,5))
    x = np.arange(10); w = 0.08
    for c in range(num_classes):
        ax.bar(x + (c - num_classes/2)*w, P[:,c], w, alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(x, U, 'ko-', label="uncertainty")

    ax.set_xticks(x)
    ax.set_xticklabels([f"s{i+1}" for i in x], rotation=45)
    ax.set_xlabel("Sample"); ax.set_ylabel("Probability")
    ax2.set_ylabel("Uncertainty")
    ax.set_title("Class probabilities + uncertainty (10 samples)")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("bar_10_samples.png")
    plt.show()
