"""Pipeline d'inférence de NeuroScan-XAI.

Ce module est volontairement séparé de l'interface Streamlit :
- le prétraitement reproduit EXACTEMENT celui de l'entraînement (Colab) ;
- il peut être testé sans lancer le site.

Prétraitement d'entraînement : niveaux de gris -> recadrage du cerveau ->
redimensionnement 256x256 (sauvegarde jpg) -> chargement en 224x224, 3 canaux,
valeurs 0-255 (EfficientNet contient déjà sa normalisation : pas de /255).
"""
import json
from pathlib import Path

import cv2
import numpy as np

NOMS_FR = {
    "glioma": "Gliome",
    "meningioma": "Méningiome",
    "notumor": "Pas de tumeur",
    "pituitary": "Tumeur de l'hypophyse",
}

T_MIN, T_MAX = 0.3, 5.0   # plage plausible d'une température de calibration


# ---------------------------------------------------------------- prétraitement
def recadrer(img: np.ndarray) -> np.ndarray:
    """Retire les bords noirs autour du cerveau (image 2D en niveaux de gris)."""
    flou = cv2.GaussianBlur(img, (5, 5), 0)
    _, masque = cv2.threshold(flou, 20, 255, cv2.THRESH_BINARY)
    masque = cv2.erode(masque, None, iterations=2)
    masque = cv2.dilate(masque, None, iterations=2)
    contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    if w * h < 0.2 * img.shape[0] * img.shape[1]:   # recadrage douteux : on garde l'original
        return img
    return img[y:y + h, x:x + w]


def preparer(pil_img, taille: int = 224):
    """PIL -> (tenseur (1, taille, taille, 3) en float32 0-255, image grise affichable)."""
    gris = np.array(pil_img.convert("L"))
    gris = cv2.resize(recadrer(gris), (256, 256))
    gris = cv2.resize(gris, (taille, taille))
    rgb = np.stack([gris] * 3, axis=-1).astype("float32")   # pas de division par 255
    return rgb[None, ...], gris


# ------------------------------------------------------------------ calibration
def temperature_valide(T) -> bool:
    return T is not None and T_MIN <= float(T) <= T_MAX


def calibrer(probs: np.ndarray, T: float) -> np.ndarray:
    """Temperature scaling appliqué à des probabilités softmax."""
    logits = np.log(np.asarray(probs, dtype="float64") + 1e-12) / float(T)
    e = np.exp(logits - logits.max())
    return e / e.sum()


# ---------------------------------------------------------------- configuration
def charger_config(chemin) -> dict:
    with open(chemin, encoding="utf-8") as f:
        cfg = json.load(f)
    for cle in ("class_names", "temperature", "img_size"):
        if cle not in cfg:
            raise KeyError(f"Clé manquante dans la configuration : {cle}")
    return cfg


# ------------------------------------------------------------------------ modèle
def charger_modele(chemin_modele, couche_cam: str = "top_activation"):
    """Charge le modèle Keras et construit le modèle auxiliaire pour Grad-CAM."""
    import tensorflow as tf   # import tardif : les tests de prétraitement n'en ont pas besoin

    model = tf.keras.models.load_model(str(chemin_modele), compile=False)
    couche = model.get_layer(couche_cam)
    grad_model = tf.keras.Model(model.input, [couche.output, model.output])
    return model, grad_model


def predire(model, x: np.ndarray, T: float = 1.0) -> np.ndarray:
    probs = np.asarray(model(x, training=False))[0]
    return calibrer(probs, T)


def gradcam(grad_model, x: np.ndarray, taille: int = 224) -> np.ndarray:
    """Carte Grad-CAM (valeurs 0-1, taille x taille) pour la classe prédite."""
    import tensorflow as tf

    x = tf.convert_to_tensor(x)
    with tf.GradientTape() as tape:
        cartes, preds = grad_model(x, training=False)
        score = preds[:, int(tf.argmax(preds[0]))]
    grads = tape.gradient(score, cartes)
    poids = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(cartes * poids[:, None, None, :], axis=-1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    return cv2.resize(cam, (taille, taille))


def superposer(gris: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpose la carte de chaleur (palette jet) sur l'image grise -> RGB uint8."""
    cam8 = np.uint8(np.clip(cam, 0, 1) * 255)
    couleur = cv2.cvtColor(cv2.applyColorMap(cam8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    base = np.stack([gris] * 3, axis=-1)
    return (base * (1 - alpha) + couleur * alpha).astype("uint8")
