import os
import torch
import torch.optim as optim

# 1) Se manca il file preprocessato, lo generiamo
if not os.path.exists("preprocessed_bearing_data.npz"):
    print("⚙️  preprocessed_bearing_data.npz non trovato, avvio preprocessing…")
    from preprocess_data import preprocess_and_save
    preprocess_and_save()

# 2) Import dei moduli
from data_utils       import load_dataset, prepare_dataloaders
from metrics_utils    import compute_metrics
from train_utils      import train_epoch
from plot_utils       import plot_training_curves, plot_sample_bar
from model_evidential import EvidentialVGG

def main():
    # 3) Carica X,y,classes
    X, y, class_names = load_dataset("preprocessed_bearing_data.npz")

    # 4) DataLoader
    train_loader, test_loader, num_classes = prepare_dataloaders(
        X, y, class_names, batch_size=32
    )

    # 5) Device, modello e ottimizzatore
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = EvidentialVGG(num_classes=num_classes, input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # 6) Storico metriche
    train_hist = {k: [] for k in ["accuracy","corr_evd","mis_evd","corr_unc","mis_unc"]}
    test_hist  = {k: [] for k in train_hist}

    # 7) Loop su 50 epoche
    num_epochs = 50
    for epoch in range(1, num_epochs+1):
        lambda_coef = 0.001 * min(1.0, epoch / 10.0)

        train_epoch(
            model, train_loader, optimizer, device, lambda_coef,
            {"risk_weight":1.0, "target_uncertainty":0.06}
        )

        t_metrics = compute_metrics(model, train_loader, device, num_classes)
        v_metrics = compute_metrics(model, test_loader,  device, num_classes)

        for key, val in zip(train_hist, t_metrics):
            train_hist[key].append(val)
        for key, val in zip(test_hist,  v_metrics):
            test_hist[key].append(val)

        print(f"Epoch {epoch:2d}/{num_epochs}  "
              f"TrainAcc={train_hist['accuracy'][-1]:.3f}  "
              f"TestAcc={test_hist['accuracy'][-1]:.3f}")

    # 8) Salva + plot
    torch.save(model.state_dict(), "evidential_cnn_model.pth")
    plot_training_curves(train_hist, test_hist)

    # 9) Bar chart 10 random sample
    X_full = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    plot_sample_bar(model, X_full, device, num_classes)

if __name__ == "__main__":
    main()
