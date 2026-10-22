# === À COLLER DANS COLAB, APRÈS la cellule de calibration ===
# Prépare UN SEUL zip (modèle + config + images d'exemple) à télécharger,
# puis à décompresser dans le dossier du projet neuroscan-xai/.
import os, json, shutil
from google.colab import drive, files
drive.mount('/content/drive')

DRIVE = "/content/drive/MyDrive"
EXPORT = "/content/neuroscan_assets"

cfg = json.load(open(f"{DRIVE}/neuroscan_config.json"))
assert 0.3 <= cfg["temperature"] <= 5.0, f"Température invalide : {cfg['temperature']}"

shutil.rmtree(EXPORT, ignore_errors=True)
os.makedirs(f"{EXPORT}/model"); os.makedirs(f"{EXPORT}/exemples")
shutil.copy(f"{DRIVE}/neuroscan_effnet.keras", f"{EXPORT}/model/")
shutil.copy(f"{DRIVE}/neuroscan_config.json", f"{EXPORT}/model/")
for f in os.listdir(f"{DRIVE}/exemples"):
    shutil.copy(f"{DRIVE}/exemples/{f}", f"{EXPORT}/exemples/")

shutil.make_archive("/content/neuroscan_assets", "zip", EXPORT)
print("Contenu :", cfg)
files.download("/content/neuroscan_assets.zip")
