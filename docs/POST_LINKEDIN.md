# Brouillon de post LinkedIn

> Remplace les éléments entre crochets. N'ajoute aucun chiffre que tu n'as pas mesuré toi-même.

---

🧠 **Mon premier modèle d'IA médicale s'est trompé… et c'est ce qui m'a le plus appris.**

En 1ʳᵉ année d'ingénierie GIIA, j'ai construit **NeuroScan-XAI** : un outil d'aide au tri d'IRM cérébrales qui classe 4 catégories (gliome, méningiome, tumeur de l'hypophyse, absence de tumeur).

Mon premier CNN, fait from scratch, atteignait **65,5 %** d'exactitude. Pire : avec Grad-CAM, j'ai vu qu'il regardait surtout le **contour du crâne**, pas la lésion.

Ce que j'ai changé :
🔧 recadrage automatique du cerveau
🔧 transfer learning avec EfficientNetB0 (fine-tuning en 2 phases)
🔧 évaluation centrée sur le recall et l'AUC, pas seulement l'accuracy

📊 Résultat sur 1 600 images de test : **86,8 % d'exactitude** (+21 points), AUC de 0,95 à 0,99 selon la classe. Pour le tri « tumeur / pas de tumeur » : environ 98 % de sensibilité. En médecine, rater une tumeur coûte plus cher qu'une fausse alerte.

🧭 L'app affiche aussi sa confiance, signale les **cas incertains** (« revue humaine requise ») et montre une carte Grad-CAM.

⚠️ Limites assumées : un seul jeu de données, pas de validation externe, risque de doublons entre train et test, gliomes et méningiomes encore confondus. C'est un projet pédagogique, pas un dispositif médical. L'IA assiste, le médecin décide.

⚙️ Stack : TensorFlow/Keras · EfficientNetB0 · OpenCV · Grad-CAM · Streamlit · GitHub

👉 **Testez la démo** : [lien Streamlit] (identifiants : [identifiant] / [mot de passe])
💻 **Code et rapport** : [lien GitHub]

Vos retours sont les bienvenus, surtout si vous travaillez en imagerie médicale 🙏

#DeepLearning #ComputerVision #IA #HealthTech #TensorFlow #Streamlit #ImagerieMédicale
