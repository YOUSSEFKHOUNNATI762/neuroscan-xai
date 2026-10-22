"""NeuroScan-XAI : aide au tri d'IRM cérébrales (démo pédagogique)."""
import hmac
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

import inference as inf

BASE = Path(__file__).parent
CHEMIN_MODELE = BASE / "model" / "neuroscan_effnet.keras"
CHEMIN_CONFIG = BASE / "model" / "neuroscan_config.json"
DOSSIER_EXEMPLES = BASE / "exemples"

st.set_page_config(page_title="NeuroScan-XAI", page_icon="🧠", layout="wide")


# ------------------------------------------------------------------ connexion
def lire_identifiants():
    try:
        return st.secrets["APP_USER"], st.secrets["APP_PASSWORD"]
    except Exception:
        return None


def identifiants_ok(user: str, pwd: str, attendus) -> bool:
    return (hmac.compare_digest(user.encode(), attendus[0].encode())
            and hmac.compare_digest(pwd.encode(), attendus[1].encode()))


if not st.session_state.get("ok"):
    st.title("🧠 NeuroScan-XAI")
    st.subheader("Cabinet virtuel : connexion")
    attendus = lire_identifiants()
    if attendus is None:
        st.error("Identifiants non configurés. Ajoutez APP_USER et APP_PASSWORD "
                 "dans les « Secrets » de l'application (voir le guide de déploiement).")
        st.stop()
    with st.form("login"):
        user = st.text_input("Identifiant")
        pwd = st.text_input("Mot de passe", type="password")
        envoye = st.form_submit_button("Se connecter")
    if envoye:
        if identifiants_ok(user, pwd, attendus):
            st.session_state["ok"] = True
            st.rerun()
        else:
            st.error("Identifiant ou mot de passe incorrect.")
    st.stop()


# --------------------------------------------------------------------- modèle
if not CHEMIN_MODELE.exists() or not CHEMIN_CONFIG.exists():
    st.error("Fichiers du modèle introuvables. Placez `neuroscan_effnet.keras` et "
             "`neuroscan_config.json` dans le dossier `model/` (voir le README).")
    st.stop()


@st.cache_resource(show_spinner="Chargement du modèle…")
def charger():
    cfg = inf.charger_config(CHEMIN_CONFIG)
    model, grad_model = inf.charger_modele(CHEMIN_MODELE)
    return model, grad_model, cfg


model, grad_model, cfg = charger()
CLASSES, TAILLE = cfg["class_names"], int(cfg["img_size"])
T = float(cfg["temperature"])
if not inf.temperature_valide(T):
    st.warning(f"Température de calibration suspecte (T = {T:.2f}) : "
               "calibration désactivée (T = 1). Recalculez T dans le notebook.")
    T = 1.0


# ------------------------------------------------------------------ interface
st.title("🧠 NeuroScan-XAI : aide au tri d'IRM cérébrales")
st.warning("Outil pédagogique pouvant commettre des erreurs. Il ne remplace pas "
           "l'avis d'un médecin et ne pose aucun diagnostic. "
           "Utilisez uniquement des images issues de jeux de données publics, "
           "jamais de vraies données de patients.")

seuil = st.sidebar.slider("Seuil d'incertitude", 0.50, 0.95, 0.70, 0.05)
alpha = st.sidebar.slider("Opacité Grad-CAM", 0.0, 1.0, 0.45, 0.05)
if st.sidebar.button("Se déconnecter"):
    st.session_state["ok"] = False
    st.rerun()

exemples = sorted(p for p in DOSSIER_EXEMPLES.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
choix = st.selectbox("Image d'exemple (optionnel)", ["(aucune)"] + [p.name for p in exemples])
fichier = st.file_uploader("ou charger une IRM (jpg, png)", type=["jpg", "jpeg", "png"])

image = None
if fichier is not None:
    image = Image.open(fichier)
elif choix != "(aucune)":
    image = Image.open(DOSSIER_EXEMPLES / choix)

if image is not None:
    x, gris = inf.preparer(image, TAILLE)
    probs = inf.predire(model, x, T)
    i = int(np.argmax(probs))
    conf = float(probs[i])
    cam = inf.gradcam(grad_model, x, TAILLE)

    c1, c2 = st.columns(2)
    c1.image(gris, caption="IRM prétraitée", width="stretch", clamp=True)
    c2.image(inf.superposer(gris, cam, alpha),
             caption="Grad-CAM : zones ayant influencé la décision", width="stretch")

    if conf < seuil:
        st.error(f"⚠️ Cas incertain (confiance {conf:.0%}) : revue humaine requise.")
    else:
        st.success(f"Estimation du modèle : **{inf.NOMS_FR.get(CLASSES[i], CLASSES[i])}** "
                   f"(confiance {conf:.0%})")

    st.bar_chart({inf.NOMS_FR.get(c, c): float(p) for c, p in zip(CLASSES, probs)})

with st.expander("À propos et limites"):
    st.markdown(
        "- Modèle : EfficientNetB0 (transfer learning), entraîné sur le *Brain Tumor MRI Dataset* "
        "(Kaggle, licence CC0).\n"
        "- Performance sur le jeu de test du dataset : environ 87 % d'exactitude. "
        "Aucune validation externe ni clinique n'a été réalisée.\n"
        "- Le Grad-CAM est un indice visuel approximatif : il ne localise pas précisément la lésion.\n"
        "- Les gliomes et méningiomes sont encore parfois confondus.\n"
        "- Ce projet n'est pas un dispositif médical."
    )
