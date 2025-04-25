import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def load_dataset(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X  = data["X_2D"]            # (N,28,28,1)
    y  = data["y_encoded"]       # (N,)
    cls= data["class_names"].tolist()
    return X, y, cls

def prepare_dataloaders(X, y, class_names, batch_size=32, test_size=0.3, random_state=42):
    num_classes = len(class_names)
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=test_size,
                                          random_state=random_state, shuffle=True)
    # to torch
    Xtr = torch.tensor(Xtr, dtype=torch.float32).permute(0,3,1,2)
    Xte = torch.tensor(Xte, dtype=torch.float32).permute(0,3,1,2)
    Ytr = F.one_hot(torch.tensor(ytr), num_classes=num_classes).float()
    Yte = F.one_hot(torch.tensor(yte), num_classes=num_classes).float()

    train_loader = DataLoader(TensorDataset(Xtr,Ytr),
                              batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(TensorDataset(Xte,Yte),
                              batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader, num_classes
