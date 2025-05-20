# preprocess_data.py

import os
import numpy as np
from scipy.io import loadmat
from scipy.signal import spectrogram

MAT_DIR = "bearing_data"
FILES_LABELS = [
    ("Dati_normali_HP(1).mat", "Normal"),
    # ...
]

def extract_de_signal(path):
    mat = loadmat(path)
    # prendi il primo array 1-D che trovi
    for k,v in mat.items():
        if isinstance(v, np.ndarray) and v.ndim==2 and v.shape[1]==1:
            return v.flatten()
    raise RuntimeError(f"No signal in {path}")

def stft_to_image(sig, nperseg=64, noverlap=32, out_size=(28,28)):
    # calcola spettrogramma
    f, t, Sxx = spectrogram(sig, fs=12_000, nperseg=nperseg, noverlap=noverlap)
    # prendi la parte log |Sxx|
    img = np.log1p(Sxx)
    # ridimensiona con semplice down‐/up‐sampling
    img_resized = np.array(
        [ np.interp(np.linspace(0, img.shape[1]-1, out_size[1]), 
                    np.arange(img.shape[1]), row)
          for row in img ])
    # vuoi 28 righe × 28 colonne, quindi riduci le freq
    img_resized = img_resized[:out_size[0], :]
    # normalizza 0–1
    img_resized = (img_resized - img_resized.min()) / (img_resized.ptp() + 1e-8)
    return img_resized

def preprocess_and_save():
    signals, labels = [], []
    for fname, lbl in FILES_LABELS:
        path = os.path.join(MAT_DIR, fname)
        sig = extract_de_signal(path)
        # segmenta a finestre di 784 point con overlap 90%
        window, step = 784, 78
        for i in range(0, len(sig)-window+1, step):
            seg = sig[i:i+window]
            img = stft_to_image(seg)
            signals.append(img)
            labels.append(lbl)
    X = np.stack(signals)         # (N,28,28)
    X = X[..., None]              # (N,28,28,1)
    class_names = sorted(set(labels))
    y = np.array([class_names.index(l) for l in labels])
    np.savez("preprocessed_bearing_data.npz",
             X_2D = X.astype(np.float32),
             y_encoded = y,
             class_names = np.array(class_names))
    print("💾 saved preprocessed_bearing_data.npz")
