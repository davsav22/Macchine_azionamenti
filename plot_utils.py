import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_training_curves(train_hist, test_hist):
    epochs = np.arange(1, len(train_hist["accuracy"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (a) Training evidence
    ax = axes[0, 0]
    ax.plot(epochs, train_hist["corr_evd"], color='r', label="correct")
    ax.plot(epochs, train_hist["mis_evd"], color='k', label="misclass")
    ax.set_title("(a) Training evidence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Estimated total evidence")
    ax.set_ylim(0, 300)
    ax.set_yticks(np.arange(0, 301, 50))
    ax.legend()

    # (b) Training acc & unc
    ax = axes[0, 1]
    ax.plot(epochs, train_hist["accuracy"], color='b', label="accuracy")
    ax.plot(epochs, train_hist["corr_unc"], color='r', linestyle='--', label="unc corr")
    ax.plot(epochs, train_hist["mis_unc"], color='k', linestyle='--', label="unc mis")
    ax.set_title("(b) Training acc & unc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.legend()

    # (c) Testing evidence
    ax = axes[1, 0]
    ax.plot(epochs, test_hist["corr_evd"], color='r', label="correct")
    ax.plot(epochs, test_hist["mis_evd"], color='k', label="misclass")
    ax.set_title("(c) Testing evidence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Estimated total evidence")
    ax.set_ylim(0, 300)
    ax.set_yticks(np.arange(0, 301, 50))
    ax.legend()

    # (d) Testing uncertainty
    ax = axes[1, 1]
    ax.plot(epochs, test_hist["corr_unc"], color='r', linestyle='--', label="unc corr")
    ax.plot(epochs, test_hist["mis_unc"], color='k', linestyle='--', label="unc mis")
    ax.set_title("(d) Testing uncertainty")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Uncertainty")
    ax.set_ylim(0, 0.8)
    ax.set_yticks(np.linspace(0, 0.8, 5))
    ax.legend()

    plt.tight_layout()
    plt.savefig("article_style_evidence_uncertainty.png", dpi=200)
    plt.show()

# Il bar‐chart delle 10 sample può restare invariato
def plot_sample_bar(model, X_full, device, num_classes):
    model.eval()
    idx = torch.randperm(X_full.size(0))[:10]
    X10 = X_full[idx].to(device)
    with torch.no_grad():
        ev    = model(X10)
        alpha = ev + 1
        S     = alpha.sum(dim=1, keepdim=True)
        P     = (alpha/S).cpu().numpy()
        U     = (num_classes/(S+num_classes)).squeeze(1).cpu().numpy()

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
    plt.tight_layout()
    plt.savefig("bar_10_samples.png", dpi=150)
    plt.show()
