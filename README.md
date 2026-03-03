🤖 Module IA : Analyseur de CIN (Résidanat TN)
Ce module utilise l'Intelligence Artificielle pour vérifier la conformité d'une Carte d'Identité Nationale tunisienne en comparant l'image avec les données d'un formulaire.

📋 Fonctionnalités
Filtre de Qualité : Détecte si l'image est trop floue.

Auto-Rotation : Redresse automatiquement la carte si elle est de travers.

Lecture OCR : Extraction du texte en Arabe et des chiffres.

Translittération Phonétique : Convertit le français en arabe pour comparer les noms (ex: slimen -> سليمان).

Système de Score : 50% pour le CIN, 50% pour les données textuelles.

🛠️ Installation
Ton ami doit suivre ces étapes dans son terminal :

1. Cloner le projet ou copier les fichiers
S'assurer que le fichier analyseur_ia.py et l'image de test sont dans le même dossier.

2. Installer les bibliothèques Python
Bash
pip install opencv-python easyocr fuzzywuzzy python-Levenshtein requests
🚀 Utilisation
Placer une photo de CIN dans le dossier (ex: ma_carte.jpg).

Modifier la zone de test en bas du fichier analyseur_ia.py avec les infos à vérifier :

Python
donnees_candidat = {
    "cin": "15354368",
    "nom": "boulabiar",
    "prenom": "syrine",
    "lieu_naissance": "slimen"
}
Lancer le script :

Bash
python analyseur_ia.py
⚠️ Notes importantes
Internet : Une connexion est nécessaire pour la première exécution (téléchargement des modèles EasyOCR) et pour la translittération des noms.

GPU : Le code est configuré par défaut sur gpu=False pour fonctionner sur tous les PC (CPU).

Fichiers .pyc : Tu peux ignorer les fichiers dans __pycache__, ils sont générés automatiquement par Python.
