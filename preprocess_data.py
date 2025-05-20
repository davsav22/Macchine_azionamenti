import os
import numpy as np
from scipy.io import loadmat

MAT_DIR = "bearing_data"
FILES_LABELS = [
    ("Dati_normali_HP(1).mat",    "Normal"),
    ("o.oo7_ball_HP(1).mat",      "Ball_0.007"),
    ("o.oo7_inner_race_HP(1).mat","IR_0.007"),
    ("o.oo7_outer_race_HP(1).mat","OR_0.007"),
    ("o.o14_ball_HP(1).mat",      "Ball_0.014"),
    ("o.o14_inner_race_HP(1).mat","IR_0.014"),
    ("o.o14_outer_race_HP(1).mat","OR_0.014"),
    ("o.o21_ball_HP(1).mat",      "Ball_0.021"),
    ("o.o21_inner_race_HP(1).mat","IR_0.021"),
    ("o.o21_outer_race_HP(1).mat","OR_0.021"),
]

def extract_signal(path):
    data = loadmat(path)
    keys = [k for k in data if not k.startswith("__")]
    for k in keys:
        if "DE_time" in k:
            return data[k].flatten()
    for k in keys:
        arr = data[k]
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr[:,0]
    raise RuntimeError(f"No signal in {path}")

def segment(signals, labels, window=784, overlap=684):
    Xs, Ys = [], []
    step = window - overlap
    for sig,lbl in zip(signals,labels):
        for i in range(0, len(sig)-window+1, step):
            Xs.append(sig[i:i+window])
            Ys.append(lbl)
    return np.array(Xs), np.array(Ys)

def to_2D(X, size=(28,28)):
    N,W = X.shape
    H,L = size
    assert H*L==W
    return X.reshape(N,H,L)[...,None]

def preprocess_and_save():
    signals, labels = [], []
    for fname,lbl in FILES_LABELS:
        p = os.path.join(MAT_DIR, fname)
        print("Loading", p)
        signals.append(extract_signal(p))
        labels.append(lbl)

    print("Segmenting…")
    X_seg,y_seg = segment(signals, labels)
    print("Reshaping…")
    X_2D = to_2D(X_seg)

    class_names = sorted(np.unique(y_seg))
    y_enc = np.array([class_names.index(l) for l in y_seg], dtype=int)

    out = "preprocessed_bearing_data.npz"
    np.savez(out,
             X_2D=X_2D.astype(np.float32),
             y_encoded=y_enc,
             class_names=np.array(class_names))
    print("Saved →", out)

if __name__=="__main__":
    preprocess_and_save()
