# Projet Chef-d'oeuvre

Une API Web permettant de prédire la réussite scolaire d'un élève.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Mise en place

Avant de lancer l'API, il faut entraîner les modèles pour générer les artefacts nécessaires (`.joblib` files) et préparer la base de données.

```bash
python3 train.py
```
Ce script va:
- Charger et nettoyer les données depuis `data/`.
- Entraîner les modèles de régression (G3) et de classification (Pass/Fail).
- Sauvegarder les modèles dans `regression_model.joblib` et `classification_model.joblib`.
- Sauvegarder les colonnes des caractéristiques dans `model_columns.joblib`.
- Enregistrer les expériences dans `mlflow.db`.

## Lancement de l'API

```bash
uvicorn app.main:app --reload
```
    
Ouvrez votre navigateur et allez à: [http://127.0.0.1:8000](http://127.0.0.1:8000)

## Fonctionnalités

-   **Prédiction:** Web form to input student data and get real-time predictions.
-   **Historique:** View past prediction logs stored in SQLite (`sql_app.db`).
-   **Données:** Preview the cleaned dataset used for training.
-   **API Docs:** Documentation Swagger disponible à [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Structure du projet

-   `app/`: Code de l'application.
    -   `main.py`: Point d'entrée et routes de FastAPI.
    -   `schemas.py`: Modèles Pydantic pour la validation des données.
    -   `models.py`: Modèles SQLAlchemy pour la base de données.
    -   `database.py`: Configuration de la base de données.
    -   `templates/`: Templates HTML pour l'interface utilisateur.
-   `data/`: Données brutes et nettoyées.
-   `train.py`: Script pour l'entraînement des modèles et le suivi avec MLFlow.
