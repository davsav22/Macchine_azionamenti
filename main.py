import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from preprocess_data import preprocess_and_save
from data_utils       import load_dataset, prepare_dataloaders
from metrics_utils    import compute_metrics
from train_utils      import train_epoch
from plot_utils       import plot_training_curves, plot_sample_bar
from model_evidential import EvidentialVGG

def main():
    # 1) preprocessing se serve
    if not os.path.exists("preprocessed_bearing_data.npz"):
        print("⚙️ preprocessing…")
        preprocess_and_save()

    # 2) carica dati e dataloader
    X, y, class_names = load_dataset("preprocessed_bearing_data.npz")
    train_loader, test_loader, num_classes = prepare_dataloaders(
        X, y, class_names, batch_size=64, test_size=0.3
    )

    # 3) modello + ottim + scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EvidentialVGG(num_classes, in_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # 4) storici
    train_hist = {k: [] for k in
                  ["accuracy","corr_evd","mis_evd","corr_unc","mis_unc"]}
    test_hist  = {k: [] for k in train_hist}

    # 5) loop epoche
    epochs = 100
    for epoch in range(1, epochs+1):
        lr = optimizer.param_groups[0]["lr"]
        print(f"\n→ Epoch {epoch}/{epochs}  LR={lr:.4e}")
        lambda_coef = 0.001 * min(1.0, epoch/10.0)

        train_epoch(model, train_loader, optimizer, device, lambda_coef,
                    {"risk_weight":1.0, "target_uncertainty":0.06})

        # step scheduler
        scheduler.step()

        # metriche
        t_met = compute_metrics(model, train_loader, device, num_classes)
        v_met = compute_metrics(model, test_loader,  device, num_classes)

        for key,val in zip(train_hist, t_met): train_hist[key].append(val)
        for key,val in zip(test_hist,  v_met): test_hist[key].append(val)

        print(f" TrainAcc={t_met[0]:.4f}  TestAcc={v_met[0]:.4f}")

    # 6) salva e plot
    torch.save(model.state_dict(), "evgg_model.pth")
    plot_training_curves(train_hist, test_hist)
    X_full = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    plot_sample_bar(model, X_full.to(device), device, num_classes)

if __name__=="__main__":
    main()
