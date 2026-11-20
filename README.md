# TD/TP 1 – Vision Artificielle (Master 2 GNIPIA)

## Contenu
- `Corrige_TD_TP_Vision_Artificielle.docx` : correction rédigée des exercices 1 à 7 et des activités 1 à 6.
- `main.py` : script Python qui réalise les traitements demandés dans le TP.
- `requirements.txt` : dépendances Python.
- `data/` : placer ici l'image de travail sous le nom `image_exemple.jpg`.
- `outputs/` : les figures et fichiers générés par le script.

## Installation
Python 3.9 ou plus récent est recommandé.

```bash
python -m venv .venv
```

Sous Windows :
```bash
.venv\Scripts\activate
```

Sous Linux/macOS :
```bash
source .venv/bin/activate
```

Installer les bibliothèques :
```bash
pip install -r requirements.txt
```

## Exécution
1. Ajouter une image autorisée dans `data/image_exemple.jpg`.
2. Lancer :
```bash
python main.py
```
3. Consulter les figures et les fichiers générés dans `outputs/`.

## Remarque sur les résultats
Les réponses numériques du TD sont calculées à partir des données de l'énoncé. Les mesures du TP (dimensions, histogrammes, pourcentage de pixels rouges et tailles des fichiers) dépendent de l'image choisie. Il faut donc exécuter le script avec l'image utilisée et commenter les résultats réellement obtenus.

## Notions travaillées
Représentation matricielle, indexation NumPy, échantillonnage, quantification, mémoire image, débit vidéo, espaces RGB/HSV, histogrammes, masque couleur, précision/rappel et compression.
