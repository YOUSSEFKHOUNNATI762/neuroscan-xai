# 🧠 NeuroScan-XAI : aide au tri d'IRM cérébrales avec IA explicable

Projet de Deep Learning (CNN) réalisé dans le cadre de ma 1ʳᵉ année d'ingénierie Génie Informatique & Intelligence Artificielle.
Un modèle classe une IRM cérébrale en 4 catégories, affiche sa confiance, signale les cas incertains et montre une carte Grad-CAM de ce qui a influencé sa décision.

> ⚠️ **Projet pédagogique, pas un dispositif médical.** L'outil peut se tromper et ne remplace pas l'avis d'un médecin. Aucune donnée de patient réel n'est utilisée.

**Démo en ligne :** _[lien Streamlit à ajouter]_ · **Identifiants de démo :** _[à ajouter]_

## Problème et objectif

Les radiologues analysent un grand nombre d'images. Un outil d'aide au **tri** peut aider à repérer plus vite les examens suspects. Le radiologue reste le décideur.
Classes : `glioma` (gliome), `meningioma` (méningiome), `pituitary` (tumeur de l'hypophyse), `notumor` (pas de tumeur).

## Données

[Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) (Masoud Nickparvar, Kaggle, licence CC0), compilation de trois jeux de données (Figshare, SARTAJ, Br35H).
Version utilisée : 5 600 images d'entraînement (1 400 par classe, dont 15 % pour la validation) et 1 600 images de test (400 par classe).

## Méthode

1. **Baseline** : CNN from scratch (4 blocs Conv + ReLU + MaxPooling, Flatten, Dense, softmax), construit en cohérence avec les formules du cours (convolution, ReLU, pooling, softmax, entropie croisée, descente de gradient).
2. **Prétraitement** : recadrage automatique du cerveau (suppression des bords noirs), redimensionnement, 3 canaux.
3. **Transfer learning** : EfficientNetB0 pré-entraîné ImageNet, en deux phases : (1) tête seule, backbone gelé ; (2) fine-tuning des 40 dernières couches avec un learning rate faible (3e-5), couches BatchNorm gelées.
4. **Augmentation** : rotation légère, zoom, translation, contraste (pas de flip vertical ni de déformation agressive).
5. **Explicabilité** : Grad-CAM.
6. **Incertitude** : seuil d'abstention (« cas incertain, revue humaine requise ») et temperature scaling de la confiance.

## Résultats (jeu de test, 1 600 images)

| Modèle | Accuracy | F1 macro |
|---|---|---|
| CNN from scratch (baseline) | 65,5 % | 0,639 |
| **EfficientNetB0 (transfer learning)** | **86,8 %** | **0,865** |

Détail par classe, EfficientNetB0 :

| Classe | Précision | Recall | F1 | AUC |
|---|---|---|---|---|
| Gliome | 0,931 | 0,740 | 0,825 | 0,961 |
| Méningiome | 0,791 | 0,757 | 0,774 | 0,954 |
| Pas de tumeur | 0,952 | 0,988 | 0,969 | 0,998 |
| Hypophyse | 0,816 | 0,988 | 0,894 | 0,996 |

**Tâche de tri « tumeur / pas de tumeur »** (calculée à partir de la matrice de confusion) : 20 tumeurs sur 1 200 classées « pas de tumeur » (sensibilité ≈ 98,3 %), 5 cerveaux sains sur 400 signalés comme tumeur (spécificité ≈ 98,8 %).
**Calibration :** erreur de calibration (ECE) brute de 0,042 avant temperature scaling.

**Points faibles :** les gliomes et les méningiomes restent parfois confondus (74 gliomes classés méningiomes ; 71 méningiomes classés hypophyse).

### Ce que j'ai appris en diagnostiquant le modèle de base
Le Grad-CAM de la baseline montrait que le réseau regardait surtout le **contour du crâne** plutôt que la lésion (apprentissage par raccourci). J'ai ajouté le recadrage du cerveau et le transfer learning, ce qui a fait passer l'exactitude de 65,5 % à 86,8 %. Les cartes Grad-CAM d'EfficientNet sont plus centrées sur le cerveau mais **ne ciblent pas toujours la tumeur** : elles restent un indice, pas une preuve.

## Limites et éthique

- Images 2D, un seul jeu de données, **pas de validation externe ni clinique**.
- Le jeu de données peut contenir des doublons et des images d'un même patient réparties entre entraînement et test : les performances peuvent être **surestimées**. Ce point n'a pas été vérifié.
- Grad-CAM : carte grossière (7×7), non localisante.
- Poids finaux : fin de l'époque 5 du fine-tuning (exactitude de validation 90,95 %). Le fine-tuning a été interrompu manuellement à l'époque 6.
- Ne jamais utiliser de vraies données de santé dans la démo (RGPD).

## Structure du dépôt

```
├── app.py                  # interface Streamlit (connexion, upload, prédiction, Grad-CAM)
├── inference.py            # pipeline d'inférence (prétraitement identique à l'entraînement)
├── requirements.txt
├── model/                  # neuroscan_effnet.keras + neuroscan_config.json
├── exemples/               # 4 images d'exemple du jeu de test
├── notebooks/              # export du notebook d'entraînement (Colab)
├── colab/                  # cellules d'export et de calibration
├── tests/                  # tests unitaires du prétraitement et de la calibration
└── docs/                   # guide de déploiement, brouillon de post LinkedIn
```

## Lancer en local

```bash
pip install -r requirements.txt
mkdir -p .streamlit && cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # puis éditer
streamlit run app.py
python -m pytest tests        # tests unitaires (pip install pytest)
```

## Technologies
Python · TensorFlow/Keras · EfficientNetB0 · OpenCV · Grad-CAM · Streamlit · Git/GitHub

## Références
- He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
- Tan & Le, *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019.
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017.
- Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
- Raghu et al., *Transfusion: Understanding Transfer Learning for Medical Imaging*, NeurIPS 2019.

_(Vérifie titres, années et DOI sur Google Scholar avant de citer.)_

## Licence des données
Brain Tumor MRI Dataset, CC0 (domaine public). Citer : « Brain Tumor MRI Dataset », Masoud Nickparvar, Kaggle.
