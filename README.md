# README - Projet de Veille sur les Séries Temporelles et Prédiction de la Consommation Électrique

Ce fichier README vise à fournir des informations et des instructions pour le projet de veille sur les séries temporelles et la prédiction de la consommation électrique des Hauts de France.

## Objectif du Projet
L'objectif principal de ce projet est de réaliser une veille sur les séries temporelles en utilisant des données de consommation électrique quotidienne brute régionale. Le projet se décompose en plusieurs étapes, notamment la découverte des données, l'analyse des séries temporelles, la désaisonnalisation, la recherche de modèles, et enfin, l'implémentation des meilleurs modèles de prévision.

## Données du Projet
Les données de consommation électrique des Hauts de France sont disponibles sur le site [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/consommation-quotidienne-brute-regionale/).

## Ressources
Pour mener à bien ce projet, vous pouvez utiliser les ressources suivantes :
- [Découverte des séries temporelles](https://drive.google.com/drive/folders/1MDKoX3FVXQx2Qax8eCRjPfwW5P1xb8pP)
- [Analyse d'une série temporelle](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)
- [Prédiction d'une série temporelle](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)

## Exigences Partielles
Le projet comporte également des exigences partielles, qui sont disponibles dans ce [tableur Google Sheets](https://docs.google.com/spreadsheets/d/1Ezj9QcUIDSMzEuNYaIxbpJIFsZWWFUamJp4lsxG6fY4/edit#gid=0).

## Planning
- Durée du projet : 7 jours (peut être plus court avec une journée de SI)
- Organisation : Individuelle

## Programme du Projet
Le projet se divise en plusieurs étapes principales :

### 1. Découverte des Séries Temporelles et Analyse
- Afficher la série temporelle de la consommation électrique en utilisant Pandas et Matplotlib.
- Agréger la série temporelle par année, trimestre, mois, semaine et jour.
- Créer une moyenne mobile sur une semaine centrée et symétrique.
- Observer la saisonnalité à travers des box-plots trimestriels, mensuels, hebdomadaires et journaliers.

### 2. Stationnarité et Tests Statistiques
- Comprendre le concept de stationnarité.
- Réaliser un test de Dickey-Fuller et un test KPSS pour évaluer la stationnarité de la série temporelle.
- Désaisonnaliser la composante annuelle de la série et refaire les tests de stationnarité.
- Déterminer si la série est plutôt additive ou multiplicative.

### 3. Veille sur les Modèles de Séries Temporelles (2-3 jours)
- Effectuer une veille sur les modèles de séries temporelles, notamment ARMA, ARIMA, SARIMA, VARIMAX, Prophet et XGBoost.

### 4. Implémentation des Meilleurs Modèles (2 jours)
- Implémenter les meilleurs modèles, notamment ARIMA, SARIMA et XGBoost.
- Tuner les paramètres des modèles en utilisant des techniques telles que l'AIC.
- Évaluer les modèles à l'aide de graphiques de résidus et de prédictions.
- Comparer les prédictions avec les valeurs réelles.

### 5. Utilisation de Darts pour la Prévision
- Utiliser Darts pour créer des modèles de prévision, y compris le modèle de Naive Drift et le modèle saisonnier Naive.
- Comparer ces modèles avec les modèles précédemment implémentés en utilisant la métrique MAPE (Mean Absolute Percentage Error).

### 6. Bonus
- Collecter des données météo et explorer d'autres modèles tels que VARIMA.
- Créer un GitHub et une application Streamlit pour présenter le projet.

## Contributeurs
Ce projet est réalisé individuellement dans le cadre d'un contexte professionnel.

**Note**: Assurez-vous de suivre les consignes et exemples fournis dans le dossier dédié au projet. Bonne exploration des séries temporelles et bonne modélisation de la consommation électrique des Hauts de France !