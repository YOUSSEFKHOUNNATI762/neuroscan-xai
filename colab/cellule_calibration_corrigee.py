# === À COLLER DANS COLAB (remplace l'ancienne cellule de calibration) ===
# Prérequis : model_tl, val_ds, y_proba, y_true, class_names existent (EfficientNet entraîné et évalué).
# Correction : val_ds est mélangé à chaque lecture. Prédictions et étiquettes doivent être
# obtenues dans la MÊME passe, sinon elles ne correspondent plus et T part à la borne (10).
import json
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

p_val, y_val = [], []
for x, y in val_ds:
    p_val.append(model_tl(x, training=False).numpy())
    y_val.append(np.argmax(y.numpy(), axis=1))
p_val = np.concatenate(p_val)
y_val = np.concatenate(y_val)

logits_val = np.log(p_val + 1e-12)
logits_test = np.log(y_proba + 1e-12)

def nll(T):
    p = softmax(logits_val / T, axis=1)
    return -np.mean(np.log(p[np.arange(len(y_val)), y_val] + 1e-12))

T = minimize_scalar(nll, bounds=(0.1, 10), method="bounded").x
print(f"Température optimale T = {T:.3f}")

def ece(probs, labels, n_bins=10):
    conf = probs.max(1); pred = probs.argmax(1); ok = (pred == labels)
    bins = np.linspace(0, 1, n_bins + 1); e = 0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return e

p_cal = softmax(logits_test / T, axis=1)
print(f"ECE avant calibration : {ece(y_proba, y_true):.4f}")
print(f"ECE après calibration : {ece(p_cal, y_true):.4f}")

# Garde-fou : une température collée aux bornes signale un problème
assert 0.3 <= T <= 5.0, f"T = {T:.3f} suspect : ne pas exporter, renvoyer la capture."

from google.colab import drive
drive.mount('/content/drive')
with open("/content/drive/MyDrive/neuroscan_config.json", "w") as f:
    json.dump({"class_names": class_names, "temperature": float(T), "img_size": 224}, f)
print("Config sauvegardée avec T =", round(float(T), 3))
