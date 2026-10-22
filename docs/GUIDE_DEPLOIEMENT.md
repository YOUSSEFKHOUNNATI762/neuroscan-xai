# Guide de déploiement : du Colab au site en ligne

## 1. Dans Colab (session où le modèle EfficientNet est encore en mémoire)

1. Colle et exécute `colab/cellule_calibration_corrigee.py`.
   Résultat attendu : **T entre 0,5 et 2**. La cellule s'arrête toute seule si T est suspect.
   Si T colle à 10 ou à 0,1 : ne continue pas, envoie la capture.
2. Colle et exécute `colab/cellule_export_pour_github.py`.
   Ton navigateur télécharge `neuroscan_assets.zip` (modèle + config + 4 images d'exemple).

> Si ta session Colab est fermée, le modèle reste sur ton Drive (`neuroscan_effnet.keras`),
> mais la calibration a besoin de `val_ds` et `y_proba` : il faudra recharger les données.

## 2. Sur ton PC

1. Décompresse `neuroscan-xai.zip` (le projet).
2. Décompresse `neuroscan_assets.zip` **dans le même dossier** pour que `model/` et `exemples/` se remplissent.
3. Vérifie : `model/neuroscan_effnet.keras` (≈ 33 Mo) et `model/neuroscan_config.json` doivent exister.

## 3. Test local (facultatif mais recommandé)

```bash
pip install -r requirements.txt
mkdir .streamlit   # s'il n'existe pas
copy .streamlit\secrets.toml.example .streamlit\secrets.toml      # Windows (cp sur Mac/Linux)
streamlit run app.py
```

## 4. GitHub

**Attention : l'envoi par le navigateur est limité à 25 Mo par fichier, et ton modèle fait ≈ 33 Mo.**
Il faut donc utiliser Git en ligne de commande (limite 100 Mo) ou GitHub Desktop.

1. Sur github.com : **New repository**, nom `neuroscan-xai`, public, sans README (le projet en a déjà un).
2. Dans le dossier du projet (installe Git si besoin : git-scm.com) :

```bash
git init
git add .
git commit -m "NeuroScan-XAI : application Streamlit et notebook"
git branch -M main
git remote add origin https://github.com/TON_COMPTE/neuroscan-xai.git
git push -u origin main
```

3. Vérifie sur GitHub que `model/neuroscan_effnet.keras` est bien présent et que
   `.streamlit/secrets.toml` **n'est pas** dans le dépôt (seul le fichier `.example` doit y être).

## 5. Streamlit Community Cloud

1. share.streamlit.io, connexion avec GitHub, **Create app**.
2. Dépôt `neuroscan-xai`, branche `main`, fichier principal `app.py`.
3. **Advanced settings** : Python **3.12**, et dans **Secrets** :
```
APP_USER = "docteur"
APP_PASSWORD = "ton-mot-de-passe"
```
4. **Deploy**. Le premier démarrage installe TensorFlow : compte quelques minutes.
5. Si l'app plante, ouvre **Manage app** et copie les logs complets.

## 6. Checklist avant LinkedIn

- [ ] Le site s'ouvre et la connexion fonctionne.
- [ ] Les 4 images d'exemple donnent une prédiction, une confiance et un Grad-CAM.
- [ ] Une image très différente (un paysage) donne un cas incertain ou une prédiction à confiance faible. Note ce comportement dans le README (le modèle n'a pas appris à refuser les images hors sujet).
- [ ] Le lien de démo et les identifiants sont ajoutés au README.
- [ ] Tes chiffres du post correspondent à ceux du README.
