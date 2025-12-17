# Projet de Classification de Maladies des Plantes

## Vue d'ensemble
Ce projet classe les images de feuilles de plantes comme saines ou malades en utilisant un Réseau de Neurones Convolutif (Transfer Learning avec MobileNetV2). Il inclut un dashboard Streamlit pour une interaction utilisateur facile.

## Prérequis
- **Python** : Version recommandée **3.9 à 3.11**. (Note : Python 3.12+ peut avoir des problèmes de compatibilité avec TensorFlow).
- **Environnement Virtuel** : Recommandé pour éviter les conflits.

## Instructions d'installation

1. **Créer un Environnement Virtuel** (Optionnel mais recommandé) :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Installer les Dépendances** :
   ```bash
   pip install -r requirements.txt
   ```
   *Note : Si vous êtes sur Apple Silicon (M1/M2/M3), assurez-vous d'installer `tensorflow-macos` si le `tensorflow` standard échoue.*

## Workflow

### 1. Exploration des Données
Exécutez le notebook d'exploration pour comprendre le jeu de données :
```bash
jupyter notebook notebooks/01_exploration_and_processing.ipynb
```

### 2. Entraînement du Modèle
Entraînez le modèle. Cela générera `plant_disease_model.h5` et `class_indices.json`.
```bash
jupyter notebook notebooks/02_model_training.ipynb
```
*Note : Assurez-vous d'exécuter toutes les cellules pour sauvegarder le modèle et les indices.*

### 3. Lancer l'Application
Démarrez l'application Streamlit :
```bash
streamlit run app.py
```
Téléchargez une image depuis `Dataset/valid` pour tester la prédiction.

## Structure du Dossier
- `Dataset/` : Contient les images `train` (entraînement) et `valid` (validation).
- `notebooks/` : Notebooks Jupyter pour l'exploration et l'entraînement.
- `app.py` : Application Streamlit.
- `requirements.txt` : Dépendances Python.
