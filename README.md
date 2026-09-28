# README - Projet de Veille sur les Séries Temporelles et Prédiction de la Consommation Électrique

Ce fichier README vise à fournir des informations et des instructions pour le projet de veille sur les séries temporelles et la prédiction de la consommation électrique des Hauts de France.

## Objectif du Projet
L'objectif principal de ce projet est de réaliser une veille sur les séries temporelles en utilisant des données de consommation électrique quotidienne brute régionale. Le projet se décompose en plusieurs étapes, notamment la découverte des données, l'analyse des séries temporelles, la désaisonnalisation, la recherche de modèles, et enfin, l'implémentation des meilleurs modèles de prévision.

## Données du Projet
Les données de consommation électrique des Hauts de France sont disponibles sur le site [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/consommation-quotidienne-brute-regionale/).

## Installation et exécution
1. Installer les dépendances : `pip install -r requirements.txt`
2. Télécharger le fichier CSV de la consommation quotidienne brute régionale (séparateur `;`) et l'enregistrer sous `data/data.csv` à la racine du projet (les fichiers `.csv` ne sont pas versionnés).
3. Pour le notebook `4_meteo_varima`, télécharger aussi la **température quotidienne régionale** (jeu de données « Température quotidienne régionale (depuis janvier 2016) » de l'ODRE, sur [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/temperature-quotidienne-regionale-depuis-janvier-2016/)) et l'enregistrer sous `data/temperature-quotidienne-regionale.csv` (séparateur `;`).
4. Exécuter les notebooks du dossier `notebook/` dans l'ordre (`1_analys`, `3_model`, puis `4_meteo_varima`), de haut en bas. `3_model` génère les fichiers `data/hdf_dataset.csv` et `data/hdf_daily.csv`. Son exécution complète prend plusieurs minutes, à cause des recherches `auto_arima`.

## Contenu du dépôt
- `notebook/1_analys.ipynb` : découverte et analyse de la série. Agrégations, moyenne mobile, box-plots de saisonnalité, tests de stationnarité (Dickey-Fuller, KPSS), désaisonnalisation annuelle et décomposition additive ou multiplicative.
- `notebook/2_veille.ipynb` : veille sur les modèles de séries temporelles.
- `notebook/3_model.ipynb` : modélisation et prévision (ARIMA, SARIMA, Naive Drift et Naive saisonnier avec Darts, XGBoost, Prophet).
- `notebook/4_meteo_varima.ipynb` : ajout de la température moyenne quotidienne des Hauts-de-France. Lien entre température et consommation, XGBoost avec et sans température, et modèle VARIMA sur le couple (consommation, température), comparé à un ARIMA sur la consommation seule.
- `notebook/correction/` : notebooks de correction.
- `app/` : application Streamlit de présentation du projet (voir ci-dessous).
- `rapport/` : rapport de veille (formats ODT et PDF).

## Application Streamlit
L'application présente le projet de façon interactive. Pour la lancer depuis la racine du dépôt :

```bash
streamlit run app/streamlit_app.py
```

Elle utilise `data/data.csv`, ou à défaut `data/hdf_daily.csv` produit par le notebook `3_model`. On peut aussi charger un fichier CSV depuis la barre latérale. Si le fichier météo `data/temperature-quotidienne-regionale.csv` est présent (ou chargé dans la barre latérale), l'application ajoute un onglet **Météo** et deux modèles avec température. Elle comporte les onglets suivants :
- **Présentation** : objectif, démarche et chiffres clés de la série.
- **Exploration** : série journalière avec moyenne mobile réglable, agrégations (semaine, mois, trimestre, année) et box-plots de saisonnalité (mois, jour de la semaine, heure).
- **Saisonnalité et stationnarité** : décomposition additive annuelle et tests ADF et KPSS, sur la série brute et sur la série désaisonnalisée.
- **Météo** (avec le fichier de température) : consommation et température sur la période commune, corrélation, et nuage de points consommation / température avec la moyenne par degré.
- **Prévisions** : choix de la période de test (un ou deux ans) et des modèles (Naive Drift, Naive saisonnier, SARIMA avec ordres manuels ou trouvés par `auto_arima`, XGBoost, Prophet et, avec la météo, XGBoost + température et VARIMA sur le couple consommation / température), graphique des prévisions et tableau MAPE, MAE et RMSE. Les modèles avec température s'entraînent à partir de 2016 (début du fichier météo) et ne sont proposés que si la température couvre toute la période de test.

Dans l'application, toutes les prévisions sont comparées sur la consommation réelle. Les modèles qui travaillent sur la série désaisonnalisée réintègrent la saisonnalité annuelle, estimée sur la seule période d'entraînement.

## Démarche de modélisation
- **Série utilisée** : la consommation des Hauts-de-France est moyennée par jour. Les modèles ARIMA, SARIMA, Naive Drift et Naive saisonnier travaillent sur la série **désaisonnalisée** : on retire la composante saisonnière annuelle (période de 365 jours) obtenue par `seasonal_decompose`.
- **Choix des ordres** : les ordres des modèles ARIMA et SARIMA ne sont pas fixés à la main. Ils sont trouvés par `auto_arima` (librairie pmdarima), qui minimise l'AIC. Les modèles statsmodels reprennent ensuite ces ordres et la constante, et chaque cellule affiche le modèle retenu.
  - ARIMA (question 7) : recherche sur toute la série, sans séparation entraînement/test.
  - SARIMA : saisonnalité hebdomadaire (m=7), recherche sur les seules données d'entraînement.
- **Évaluation** : les modèles de prévision gardent la dernière ou les deux dernières années en test. Ils sont comparés avec la MAPE, ainsi qu'avec le MAE et le RMSE.
- **Météo et VARIMA** (`4_meteo_varima`) : la période commune aux deux fichiers commence en 2016. Le VARIMA modélise ensemble la consommation et la température désaisonnalisées. Ses ordres sont choisis automatiquement : `d` selon le test de Dickey-Fuller, `p` selon l'AIC d'un VAR, et `q = 0`.
- **XGBoost** s'entraîne sur des variables calendaires (jour de la semaine, jour de l'année, mois, trimestre, année).

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