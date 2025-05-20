import os
import torch
import torch.optim as optim
from data_utils   import load_dataset, prepare_dataloaders
from metrics_utils import compute_metrics
from train_utils  import train_epoch
from plot_utils   import plot_training_curves, plot_sample_bar
from model_evidential import EvidentialVGG

def main():
    # 1) Preprocess se serve...
    if not os.path.exists("preprocessed_bearing_data.npz"):
        from preprocess_data import preprocess_and_save
        preprocess_and_save()

    # 2) Carica dati
    X, y, class_names = load_dataset("preprocessed_bearing_data.npz")
    train_loader, test_loader, num_classes = prepare_dataloaders(X, y, class_names, batch_size=32)

    # 3) Device, modello, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model   = EvidentialVGG(num_classes=num_classes, input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4) Storage
    train_hist = {k: [] for k in ["accuracy","corr_evd","mis_evd","corr_unc","mis_unc"]}
    test_hist  = train_hist.copy()

    # 5) Loop 50 epoche
    for ep in range(1, 51):
        # λ ramp‐up: da 0→1 in 10 epoche
        lambda_coef = min(1.0, ep / 10.0)
        train_epoch(
            model, train_loader, optimizer, device, lambda_coef,
            {"risk_weight": 1.0, "target_uncertainty": 0.06}
        )

        # prende metriche
        t = compute_metrics(model, train_loader, device, num_classes)
        v = compute_metrics(model, test_loader,  device, num_classes)
        for k,val in zip(train_hist, t): train_hist[k].append(val)
        for k,val in zip(test_hist,  v): test_hist[k].append(val)

        print(f"Epoch {ep:2d}/50  TrainAcc={train_hist['accuracy'][-1]:.3f}  "
              f"TestAcc={test_hist['accuracy'][-1]:.3f}")

    # 6) Salva e plotta
    torch.save(model.state_dict(), "evidential_vgg.pth")
    plot_training_curves(train_hist, test_hist)

    # 7) Bar chart
    X_full = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    plot_sample_bar(model, X_full, device, num_classes)

if __name__=="__main__":
    main()
