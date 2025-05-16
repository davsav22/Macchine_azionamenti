import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# 1) Se manca il .npz preprocessato, lo generiamo
if not os.path.exists("preprocessed_bearing_data.npz"):
    print("⚙️  preprocessed_bearing_data.npz non trovato, avvio preprocessing...")
    from preprocess_data import preprocess_and_save
    preprocess_and_save()

# 2) Import moduli
from data_utils       import load_dataset, prepare_dataloaders
from metrics_utils    import compute_metrics
from train_utils      import train_epoch
from plot_utils       import plot_training_curves, plot_sample_bar
from model_evidential import EvidentialVGG

def main():
    # 3) Carica
    X, y, class_names = load_dataset("preprocessed_bearing_data.npz")

    # 4) Dataloaders
    train_loader, test_loader, num_classes = prepare_dataloaders(
        X, y, class_names, batch_size=64
    )

    # 5) Setup
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model     = EvidentialVGG(num_classes=num_classes, input_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # per early-stopping su mis_unc validation
    best_mis_unc = float('inf')
    patience = 5
    wait = 0

    # 6) Storici metriche
    train_hist = {k: [] for k in ['accuracy','corr_evd','mis_evd','corr_unc','mis_unc']}
    test_hist  = {k: [] for k in train_hist}

    # 7) Loop
    for epoch in range(1, 51):
        lambda_coef = 0.001 * min(1.0, epoch / 10.0)

        train_epoch(
            model, train_loader, optimizer, device, lambda_coef,
            {'risk_weight':1.0, 'target_uncertainty':0.06}
        )
        scheduler.step()

        # metrics
        t_metrics = compute_metrics(model, train_loader, device, num_classes)
        v_metrics = compute_metrics(model, test_loader,  device, num_classes)

        # registra
        for key, val in zip(train_hist, t_metrics): train_hist[key].append(val)
        for key, val in zip(test_hist,  v_metrics): test_hist[key].append(val)

        # early-stop su mis_unc
        mis_unc = test_hist['mis_unc'][-1]
        if mis_unc < best_mis_unc:
            best_mis_unc = mis_unc
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f">> Early stopping: mis_unc non migliora da {patience} epoche")
                break

        print(
            f"Epoch {epoch:2d}/50  "
            f"TrainAcc={train_hist['accuracy'][-1]:.3f}  "
            f"TestAcc={test_hist['accuracy'][-1]:.3f}  "
            f"mis_unc={mis_unc:.3f}"
        )

    # 8) Salva + plot
    torch.save(model.state_dict(), 'evidential_vgg_model.pth')
    plot_training_curves(train_hist, test_hist)

    # 9) Bar chart 10 esempi
    X_full = torch.tensor(X, dtype=torch.float32).permute(0,3,1,2)
    plot_sample_bar(model, X_full, device, num_classes)


if __name__ == '__main__':
    main()
