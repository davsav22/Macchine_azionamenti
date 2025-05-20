import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_dataset(path="preprocessed_bearing_data.npz"):
    data = np.load(path, allow_pickle=True)
    return data["X_2D"], data["y_encoded"], data["class_names"].tolist()

def prepare_dataloaders(X, y, class_names,
                        batch_size=32,
                        test_size=0.3,
                        random_state=42):
    num_classes = len(class_names)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32).permute(0,3,1,2)
    X_te = torch.tensor(X_te, dtype=torch.float32).permute(0,3,1,2)
    Y_tr = F.one_hot(torch.tensor(y_tr, dtype=torch.long), num_classes).float()
    Y_te = F.one_hot(torch.tensor(y_te, dtype=torch.long), num_classes).float()

    train_loader = DataLoader(
        TensorDataset(X_tr, Y_tr),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        TensorDataset(X_te, Y_te),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    return train_loader, test_loader, num_classes
