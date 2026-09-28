# Prévision de la consommation électrique des Hauts-de-France

Projet de veille sur les **séries temporelles** : analyse et prévision de la consommation électrique brute quotidienne des Hauts-de-France (données RTE), avec des modèles statistiques (ARIMA, SARIMA, VARIMA), des modèles de référence (Naive Drift, Naive saisonnier) et des modèles d'apprentissage automatique (XGBoost, Prophet). Une application Streamlit présente l'ensemble de façon interactive.

## Contenu du dépôt

```
├── notebook/
│   ├── 1_analys.ipynb          # analyse de la série : saisonnalités, stationnarité, décomposition
│   ├── 2_veille.ipynb          # veille sur les modèles de séries temporelles
│   ├── 3_model.ipynb           # modélisation et prévision
│   ├── 4_meteo_varima.ipynb    # apport de la température, modèle VARIMA
│   └── correction/             # notebooks de correction
├── app/
│   ├── streamlit_app.py        # application Streamlit
│   ├── donnees.py              # chargement et préparation des données
│   └── modeles.py              # modèles de prévision utilisés par l'application
├── rapport/                    # rapport de veille (ODT et PDF)
├── requirements.txt
└── LICENSE
```

## Données

Les fichiers de données ne sont pas versionnés (le `.gitignore` exclut les `.csv`). Ils se placent dans un dossier `data/` à la racine du dépôt.

| Fichier | Contenu | Source | Utilisé par |
|---|---|---|---|
| `data/data.csv` | Consommation brute régionale au pas de 30 minutes (séparateur `;`) | [Consommation quotidienne brute régionale](https://www.data.gouv.fr/fr/datasets/consommation-quotidienne-brute-regionale/), data.gouv.fr | tous les notebooks et l'application (**obligatoire**) |
| `data/temperature-quotidienne-regionale.csv` | Températures quotidiennes par région, depuis 2016 (séparateur `;`) | [Température quotidienne régionale](https://www.data.gouv.fr/fr/datasets/temperature-quotidienne-regionale-depuis-janvier-2016/) (ODRE), data.gouv.fr | `4_meteo_varima` et l'onglet Météo de l'application (facultatif) |
| `data/hdf_dataset.csv` | Consommation des Hauts-de-France au pas de 30 minutes | généré par `3_model` | `3_model` |
| `data/hdf_daily.csv` | Consommation journalière des Hauts-de-France | généré par `3_model` | `4_meteo_varima`, l'application (à défaut de `data.csv`) |

Seules les données des Hauts-de-France (code INSEE 32) sont utilisées.

## Installation

Le projet a été testé avec Python 3.11.

```bash
pip install -r requirements.txt
```

## Utilisation

### Notebooks

Exécuter les notebooks du dossier `notebook/` dans l'ordre, de haut en bas :

1. `1_analys` : analyse de la série.
2. `3_model` : modélisation. Il génère `data/hdf_dataset.csv` et `data/hdf_daily.csv`. Son exécution complète prend plusieurs minutes, à cause des recherches `auto_arima`.
3. `4_meteo_varima` : apport de la température (nécessite le fichier de température).

`3_model` et `4_meteo_varima` sont versionnés sans leurs sorties : il faut les exécuter pour voir les graphiques et les résultats.

### Application Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'application lit `data/data.csv`, ou à défaut `data/hdf_daily.csv`. Les fichiers de consommation et de température peuvent aussi être chargés depuis la barre latérale. Elle comporte les onglets suivants :

- **Présentation** : objectif, démarche et chiffres clés de la série.
- **Exploration** : série journalière avec moyenne mobile réglable, agrégations (semaine, mois, trimestre, année) et box-plots de saisonnalité (mois, jour de la semaine, heure).
- **Saisonnalité et stationnarité** : décomposition additive annuelle, tests ADF et KPSS sur la série brute et sur la série désaisonnalisée.
- **Météo** (seulement avec le fichier de température) : consommation et température sur la période commune, corrélation, nuage de points consommation / température avec la moyenne par degré.
- **Prévisions** : choix de la période de test (un ou deux ans) et des modèles, graphique des prévisions et tableau MAPE, MAE et RMSE.

Modèles disponibles dans l'onglet Prévisions :

| Modèle | Série utilisée | Remarque |
|---|---|---|
| Naive Drift | désaisonnalisée | prolonge la tendance récente |
| Naive saisonnier | désaisonnalisée | répète la dernière semaine (ou 2 ou 4 semaines) |
| SARIMA | désaisonnalisée | ordres manuels, ou trouvés par `auto_arima` (saisonnalité de 7 jours) |
| XGBoost | brute | variables calendaires |
| Prophet | brute | proposé si la librairie `prophet` est installée |
| XGBoost + température | brute | avec le fichier de température |
| VARIMA (conso + température) | désaisonnalisées | avec le fichier de température |

Pour que les MAPE soient comparables, toutes les prévisions sont évaluées sur la consommation réelle : les modèles qui travaillent sur la série désaisonnalisée réintègrent la saisonnalité annuelle, estimée sur la seule période d'entraînement. Les deux modèles avec température s'entraînent à partir de 2016 (début du fichier météo) et ne sont proposés que si la température couvre toute la période de test.

L'application affiche un avertissement quand l'optimisation du SARIMA ne converge pas ou que sa prévision est incohérente (consommations nulles, négatives ou infinies).

## Démarche

### Analyse (`1_analys`)
- Agrégations par année, trimestre, mois, semaine et jour ; moyenne mobile centrée sur une semaine.
- Saisonnalités annuelle, hebdomadaire et journalière, observées avec des box-plots (en heure de Paris).
- Tests de stationnarité de Dickey-Fuller augmenté et KPSS, avant et après désaisonnalisation.
- Décompositions additive et multiplicative avec une période annuelle (365 jours).

### Modélisation (`3_model`)
- **Série utilisée** : la consommation moyenne journalière. Les modèles ARIMA, SARIMA, Naive Drift et Naive saisonnier travaillent sur la série **désaisonnalisée** : on retire la composante saisonnière annuelle obtenue par `seasonal_decompose`.
- **Choix des ordres** : les ordres des modèles ARIMA et SARIMA sont trouvés par `auto_arima` (librairie pmdarima), qui minimise l'AIC. Les modèles statsmodels reprennent ces ordres et la constante, et chaque cellule affiche le modèle retenu.
  - ARIMA : recherche sur toute la série, sans séparation entraînement / test.
  - SARIMA : saisonnalité hebdomadaire (m = 7), recherche sur les seules données d'entraînement.
- **Darts** : Naive Drift et Naive saisonnier (K = 7) sur la série désaisonnalisée.
- **XGBoost** : variables calendaires (jour de la semaine, jour de l'année, mois, trimestre, année).
- **Prophet** : sur la série journalière brute.
- **Évaluation** : la dernière ou les deux dernières années servent de test ; les modèles sont comparés avec la MAPE, le MAE et le RMSE.

### Météo et VARIMA (`4_meteo_varima`)
- Température moyenne quotidienne des Hauts-de-France, fusionnée avec la consommation sur la période commune (à partir de 2016).
- XGBoost avec et sans température, pour mesurer l'apport de la météo.
- VARIMA sur le couple (consommation, température) désaisonnalisé, comparé à un ARIMA sur la consommation seule. Ordres choisis automatiquement : `d` selon le test de Dickey-Fuller, `p` selon l'AIC d'un VAR, `q = 0`.

## Limites

- **Température observée** : sur la période de test, les modèles avec température utilisent la température réellement mesurée, pas une prévision météo. Leur avantage est donc un maximum, qu'une vraie prévision météo réduirait.
- **Horizon long** : sur un ou deux ans, les prévisions d'un modèle autorégressif (ARIMA, SARIMA, VARIMA) convergent vite vers une valeur moyenne ; l'essentiel de la forme de la prévision vient alors de la saisonnalité annuelle réintégrée.
- **Convergence du SARIMA** : selon les données et les ordres choisis, l'optimisation peut ne pas converger. L'application le signale ; dans les notebooks, statsmodels affiche un `ConvergenceWarning`.
- **Désaisonnalisation dans les notebooks** : dans `3_model`, la composante saisonnière est estimée sur toute la série, période de test comprise. L'application et `4_meteo_varima` l'estiment sur la seule période d'entraînement.

## Contexte du projet

Projet individuel de 7 jours réalisé en formation, en six étapes :

1. Découverte et analyse de la série (agrégations, moyenne mobile, box-plots).
2. Stationnarité (tests de Dickey-Fuller et KPSS, désaisonnalisation, série additive ou multiplicative).
3. Veille sur les modèles (ARMA, ARIMA, SARIMA, VARIMAX, Prophet, XGBoost) : voir `2_veille` et le dossier `rapport/`.
4. Implémentation des meilleurs modèles, réglage des paramètres par l'AIC, analyse des résidus.
5. Prévision avec Darts (Naive Drift, Naive saisonnier) et comparaison par la MAPE.
6. Bonus : données météo, modèle VARIMA, dépôt GitHub et application Streamlit.

Ressources :
- [Découverte des séries temporelles](https://drive.google.com/drive/folders/1MDKoX3FVXQx2Qax8eCRjPfwW5P1xb8pP)
- [Analyse d'une série temporelle](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)
- [Prédiction d'une série temporelle](https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/)
- [Exigences du projet](https://docs.google.com/spreadsheets/d/1Ezj9QcUIDSMzEuNYaIxbpJIFsZWWFUamJp4lsxG6fY4/edit#gid=0) (tableur Google Sheets)

## Licence

Distribué sous licence MIT (voir [LICENSE](LICENSE)).
