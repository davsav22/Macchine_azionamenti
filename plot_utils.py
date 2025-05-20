# plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_training_curves(train_hist, test_hist):
    epochs = np.arange(1, len(train_hist["accuracy"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # --- (a) Training evidence ---
    ax = axes[0, 0]
    ax.plot(epochs, train_hist["corr_evd"], color='r', label="correct")
    ax.plot(epochs, train_hist["mis_evd"], color='k', label="misclass")
    ax.set_title("(a) Training evidence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Estimated total evidence")
    ax.set_ylim(0, 300)                       # **forza** da 0 a 300
    ax.set_yticks(np.arange(0, 301, 50))      # tacche ogni 50
    ax.legend()

    # --- (b) Training acc & unc ---
    ax = axes[0, 1]
    ax.plot(epochs, train_hist["accuracy"], color='b', label="accuracy")
    ax.plot(epochs, train_hist["corr_unc"], color='r', linestyle='--', label="unc corr")
    ax.plot(epochs, train_hist["mis_unc"], color='k', linestyle='--', label="unc mis")
    ax.set_title("(b) Training acc & unc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.0)                       # **forza** da 0 a 1.0
    ax.set_yticks(np.linspace(0, 1.0, 6))     # tacche ogni 0.2
    ax.legend()

    # --- (c) Testing evidence ---
    ax = axes[1, 0]
    ax.plot(epochs, test_hist["corr_evd"], color='r', label="correct")
    ax.plot(epochs, test_hist["mis_evd"], color='k', label="misclass")
    ax.set_title("(c) Testing evidence")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Estimated total evidence")
    ax.set_ylim(0, 300)                       # **forza** da 0 a 300
    ax.set_yticks(np.arange(0, 301, 50))
    ax.legend()

    # --- (d) Testing uncertainty ---
    ax = axes[1, 1]
    ax.plot(epochs, test_hist["corr_unc"], color='r', linestyle='--', label="unc corr")
    ax.plot(epochs, test_hist["mis_unc"], color='k', linestyle='--', label="unc mis")
    ax.set_title("(d) Testing uncertainty")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Uncertainty")
    ax.set_ylim(0, 0.8)                       # **forza** da 0 a 0.8
    ax.set_yticks(np.linspace(0, 0.8, 5))     # tacche ogni 0.2
    ax.legend()

    plt.tight_layout()
    plt.savefig("article_style_evidence_uncertainty.png", dpi=200)
    plt.show()

def plot_sample_bar(model, X_full, device, num_classes):
    model.eval()
    # ... lasciare invariato il tuo code per i bar charts ...
    # non tocca i limiti di scala di questo grafico
    pass
