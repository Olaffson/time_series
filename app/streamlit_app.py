"""Application de présentation du projet : consommation électrique des Hauts-de-France.

Lancement depuis la racine du dépôt : `streamlit run app/streamlit_app.py`
"""

import importlib.util
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

sys.path.insert(0, str(Path(__file__).resolve().parent))

import donnees  # noqa: E402
import modeles  # noqa: E402

st.set_page_config(page_title="Consommation électrique Hauts-de-France", page_icon="⚡", layout="wide")

# Couleurs fixes par entité (palette catégorielle validée, dans l'ordre) ; la série réelle reste neutre
COULEUR_REEL = "#7a7974"
COULEURS_MODELES = {
    "Naive Drift": "#2a78d6",
    "Naive saisonnier": "#eb6834",
    "SARIMA": "#1baf7a",
    "XGBoost": "#eda100",
    "Prophet": "#e87ba4",
    "XGBoost + température": "#008300",
    "VARIMA (conso + température)": "#4a3aa7",
}
MODELES_METEO = ("XGBoost + température", "VARIMA (conso + température)")
COULEUR_TEMPERATURE = "#eb6834"
MOIS = ["Janv.", "Févr.", "Mars", "Avr.", "Mai", "Juin", "Juil.", "Août", "Sept.", "Oct.", "Nov.", "Déc."]
JOURS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
UNITE = "Consommation moyenne (MW)"


# --- Données -----------------------------------------------------------------

def date_modification(contenu, nom):
    """Date de modification du fichier local : un fichier remplacé est ainsi relu, et pas pris dans le cache."""
    return Path(nom).stat().st_mtime if contenu is None else None


@st.cache_data(show_spinner="Chargement des données…")
def charger(contenu, nom, modifie_le=None):
    source = io.BytesIO(contenu) if contenu is not None else nom
    demi_horaire = donnees.serie_demi_horaire(donnees.lire_brut(source))
    journaliere = donnees.serie_journaliere(demi_horaire)
    # Un fichier déjà journalier (hdf_daily.csv) ne permet pas l'analyse par heure
    pas_horaire = demi_horaire.index.to_series().diff().median() < pd.Timedelta(days=1)
    return (demi_horaire if pas_horaire else None), journaliere


def source_des_donnees():
    st.sidebar.header("Données")
    fichier = st.sidebar.file_uploader("Fichier CSV RTE (facultatif)", type="csv")
    if fichier is not None:
        return fichier.getvalue(), fichier.name
    for nom in ("data.csv", "hdf_daily.csv"):
        chemin = donnees.DOSSIER_DATA / nom
        if chemin.exists():
            st.sidebar.caption(f"Fichier utilisé : `data/{nom}`")
            return None, str(chemin)
    return None, None


@st.cache_data(show_spinner="Chargement de la température…")
def charger_temperature(contenu, nom, modifie_le=None):
    return donnees.lire_temperature(io.BytesIO(contenu) if contenu is not None else nom)


def source_temperature():
    fichier = st.sidebar.file_uploader("Fichier météo ODRE (facultatif)", type="csv",
                                       help="Température quotidienne régionale, séparateur `;`.")
    if fichier is not None:
        return fichier.getvalue(), fichier.name
    chemin = donnees.DOSSIER_DATA / donnees.FICHIER_TEMPERATURE
    if chemin.exists():
        st.sidebar.caption(f"Météo : `data/{donnees.FICHIER_TEMPERATURE}`")
        return None, str(chemin)
    st.sidebar.caption(f"Sans fichier météo (`data/{donnees.FICHIER_TEMPERATURE}`), l'onglet Météo et les "
                       "modèles avec température sont masqués.")
    return None, None


def mise_en_forme(fig, hauteur=420):
    fig.update_layout(
        height=hauteur,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(showgrid=False)
    return fig


# --- Onglets -----------------------------------------------------------------

def onglet_presentation(journaliere):
    st.markdown(
        """
Ce projet étudie la **consommation électrique brute quotidienne des Hauts-de-France**
(données RTE publiées sur [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/consommation-quotidienne-brute-regionale/))
et compare plusieurs modèles pour la prévoir.

**Démarche**
1. **Exploration** : agrégations et saisonnalités (annuelle, hebdomadaire, journalière).
2. **Stationnarité** : tests de Dickey-Fuller et KPSS, puis retrait de la saisonnalité annuelle.
3. **Prévision** : Naive Drift, Naive saisonnier, SARIMA, XGBoost et Prophet, comparés avec la MAPE
   sur une période de test.
4. **Météo** (si le fichier de température est disponible) : lien entre température et consommation,
   XGBoost avec la température et modèle VARIMA sur le couple consommation / température.
"""
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Période couverte", f"{journaliere.index[0]:%Y} – {journaliere.index[-1]:%Y}")
    c2.metric("Nombre de jours", f"{len(journaliere):,}".replace(",", " "))
    c3.metric("Consommation moyenne", f"{journaliere.mean():,.0f} MW".replace(",", " "))
    c4.metric("Journée record", f"{journaliere.max():,.0f} MW".replace(",", " "), help=f"{journaliere.idxmax():%d/%m/%Y}")


def onglet_exploration(demi_horaire, journaliere):
    st.subheader("Série journalière et moyenne mobile")
    fenetre = st.slider("Fenêtre de la moyenne mobile centrée (jours)", 3, 365, 7, step=2)
    fig = go.Figure()
    fig.add_scatter(x=journaliere.index, y=journaliere, name="Consommation journalière",
                    line=dict(color=COULEUR_REEL, width=1), opacity=0.6)
    fig.add_scatter(x=journaliere.index, y=journaliere.rolling(fenetre, center=True).mean(),
                    name=f"Moyenne mobile sur {fenetre} jours", line=dict(color=COULEURS_MODELES["Naive Drift"], width=2))
    fig.update_yaxes(title=UNITE)
    st.plotly_chart(mise_en_forme(fig))

    st.subheader("Agrégations")
    frequences = {"Semaine": "W", "Mois": "M", "Trimestre": "Q", "Année": "Y"}
    choix = st.radio("Agréger par", list(frequences), horizontal=True)
    agregee = journaliere.resample(frequences[choix]).mean()
    fig = go.Figure(go.Scatter(x=agregee.index, y=agregee, mode="lines+markers" if len(agregee) <= 60 else "lines", name=choix,
                               line=dict(color=COULEURS_MODELES["Naive Drift"], width=2), marker=dict(size=8)))
    fig.update_yaxes(title=UNITE)
    st.plotly_chart(mise_en_forme(fig, 360))

    st.subheader("Saisonnalités")
    dimensions = ["Mois", "Jour de la semaine"] + (["Heure de la journée"] if demi_horaire is not None else [])
    dimension = st.radio("Distribution par", dimensions, horizontal=True)
    if dimension == "Mois":
        valeurs, groupes, libelles = journaliere, journaliere.index.month - 1, MOIS
    elif dimension == "Jour de la semaine":
        valeurs, groupes, libelles = journaliere, journaliere.index.dayofweek, JOURS
    else:
        horaire = demi_horaire.resample("h").mean().dropna()
        valeurs, groupes, libelles = horaire, horaire.index.hour, [f"{h:02d}h" for h in range(24)]
    fig = go.Figure()
    for g in sorted(set(groupes)):
        fig.add_box(y=valeurs[groupes == g], name=libelles[g], marker_color=COULEURS_MODELES["Naive Drift"],
                    line=dict(width=1.5), boxpoints=False, showlegend=False)
    fig.update_yaxes(title=UNITE)
    fig = mise_en_forme(fig, 380)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig)
    if demi_horaire is None:
        st.caption("La distribution par heure nécessite le fichier RTE au pas de 30 minutes (`data/data.csv`).")


def tests_stationnarite(serie):
    adf = adfuller(serie.dropna())
    with warnings.catch_warnings():
        # KPSS prévient quand la p-valeur sort de sa table (bornée entre 0,01 et 0,1)
        warnings.simplefilter("ignore", InterpolationWarning)
        _, p_kpss, *_ = kpss(serie.dropna(), regression="c", nlags="auto")
    return {
        "ADF : p-valeur": round(adf[1], 4),
        "ADF : conclusion": "stationnaire" if adf[1] <= 0.05 else "non stationnaire",
        "KPSS : p-valeur": round(p_kpss, 4),
        "KPSS : conclusion": "non stationnaire" if p_kpss <= 0.05 else "stationnaire",
    }


@st.cache_data(show_spinner="Décomposition et tests de stationnarité…")
def analyse_stationnarite(journaliere):
    decomposition = seasonal_decompose(journaliere, model="additive", period=365)
    composantes = {
        "Série observée": decomposition.observed,
        "Tendance": decomposition.trend,
        "Saisonnalité annuelle": decomposition.seasonal,
        "Résidus": decomposition.resid,
    }
    desaison, _ = donnees.desaisonnaliser(journaliere)
    tableau = pd.DataFrame(
        {"Série journalière": tests_stationnarite(journaliere), "Série désaisonnalisée": tests_stationnarite(desaison)}
    ).T
    return composantes, tableau


def onglet_stationnarite(journaliere):
    composantes, tableau = analyse_stationnarite(journaliere)
    st.subheader("Décomposition additive (période annuelle de 365 jours)")
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=list(composantes))
    for i, serie in enumerate(composantes.values(), start=1):
        fig.add_scatter(x=serie.index, y=serie, row=i, col=1, showlegend=False,
                        line=dict(color=COULEURS_MODELES["Naive Drift"], width=1.2))
    fig = mise_en_forme(fig, 720)
    fig.update_layout(hovermode="x")
    st.plotly_chart(fig)

    st.subheader("Tests de stationnarité")
    st.markdown(
        "- **Dickey-Fuller augmenté (ADF)** : H0 = la série a une racine unitaire (non stationnaire).\n"
        "- **KPSS** : H0 = la série est stationnaire.\n\n"
        "Seuil de 5 %. Les p-valeurs KPSS sont bornées entre 0,01 et 0,1 par la table utilisée."
    )
    st.dataframe(tableau)


def onglet_meteo(journaliere, temperature):
    commun = pd.concat([journaliere.rename("consommation"), temperature], axis=1, join="inner").dropna()
    if len(commun) < 30:
        st.warning("La consommation et la température n'ont presque aucune date en commun.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Période commune", f"{commun.index[0]:%Y} – {commun.index[-1]:%Y}",
              help=f"Du {commun.index[0]:%d/%m/%Y} au {commun.index[-1]:%d/%m/%Y}")
    c2.metric("Température moyenne", f"{commun['temperature'].mean():.1f} °C".replace(".", ","))
    c3.metric("Corrélation température / consommation", f"{commun.corr().iloc[0, 1]:.2f}".replace(".", ","))

    st.subheader("Consommation et température")
    # Deux graphiques alignés plutôt qu'un double axe : chaque série garde sa propre échelle
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Consommation journalière (MW)", "Température moyenne (°C)"])
    fig.add_scatter(x=commun.index, y=commun["consommation"], row=1, col=1, name="Consommation",
                    line=dict(color=COULEURS_MODELES["Naive Drift"], width=1.2), showlegend=False)
    fig.add_scatter(x=commun.index, y=commun["temperature"], row=2, col=1, name="Température",
                    line=dict(color=COULEUR_TEMPERATURE, width=1.2), showlegend=False)
    fig = mise_en_forme(fig, 560)
    fig.update_layout(hovermode="x")
    st.plotly_chart(fig)

    st.subheader("Consommation en fonction de la température")
    paliers = commun.groupby(commun["temperature"].round())["consommation"].mean()
    fig = go.Figure()
    fig.add_scatter(x=commun["temperature"], y=commun["consommation"], mode="markers", name="Jours",
                    marker=dict(color=COULEURS_MODELES["Naive Drift"], size=5, opacity=0.35),
                    hovertemplate="%{x:.1f} °C : %{y:,.0f} MW<extra></extra>")
    fig.add_scatter(x=paliers.index, y=paliers, mode="lines", name="Moyenne par degré",
                    line=dict(color=COULEUR_REEL, width=2.5))
    fig.update_xaxes(title="Température moyenne (°C)")
    fig.update_yaxes(title=UNITE)
    fig = mise_en_forme(fig, 440)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig)
    st.caption("La consommation augmente quand il fait froid, sous l'effet du chauffage électrique. "
               "La courbe des moyennes par degré montre où cet effet s'atténue.")


@st.cache_data(show_spinner="Recherche des ordres SARIMA avec auto_arima (environ une minute)…")
def ordres_auto(train):
    return modeles.ordres_auto_arima(train)


@st.cache_data(show_spinner="Entraînement du modèle…")
def prevoir(nom, train, horizon, parametres, temperature=None):
    """Prévision du modèle `nom`, et ordres retenus pour VARIMA (None pour les autres modèles)."""
    if nom in MODELES_METEO:
        # Les modèles météo s'entraînent sur la période où la température est disponible
        train = train[train.index >= temperature.index[0]]
        if nom == "XGBoost + température":
            return modeles.xgboost(train, horizon, temperature), None
        return modeles.varima(train, horizon, temperature)
    return _prevoir_sans_meteo(nom, train, horizon, parametres), None


def _prevoir_sans_meteo(nom, train, horizon, parametres):
    if nom == "Naive Drift":
        return modeles.naive_drift(train, horizon)
    if nom == "Naive saisonnier":
        return modeles.naive_saisonnier(train, horizon, k=parametres["k"])
    if nom == "SARIMA":
        return modeles.sarima(train, horizon, parametres["order"], parametres["seasonal_order"], parametres["trend"])
    if nom == "XGBoost":
        return modeles.xgboost(train, horizon)
    return modeles.prophet(train, horizon)


def prophet_disponible():
    return importlib.util.find_spec("prophet") is not None


def meteo_utilisable(train, test, temperature):
    """Les modèles météo ont besoin de la température sur toute la période et de deux ans d'entraînement."""
    if temperature is None:
        return False
    debut = max(train.index[0], temperature.index[0])
    couverture = temperature.reindex(pd.date_range(debut, test.index[-1], freq="D"))
    return couverture.notna().all() and (train.index >= debut).sum() >= 2 * modeles.PERIODE_ANNUELLE


def onglet_previsions(journaliere, temperature):
    st.markdown(
        "Les modèles s'entraînent sur le passé et prévoient la période de test. Naive Drift, Naive saisonnier "
        "et SARIMA travaillent sur la série désaisonnalisée ; la saisonnalité annuelle, estimée sur "
        "l'entraînement seul, est ensuite réintégrée. Toutes les prévisions sont donc comparées sur la "
        "consommation réelle."
    )
    c1, c2 = st.columns([1, 2])
    annees_test = c1.radio("Période de test", [1, 2], index=1, format_func=lambda a: f"{a} an" + ("s" if a > 1 else ""),
                           horizontal=True)
    horizon = 365 * annees_test
    train, test = journaliere.iloc[:-horizon], journaliere.iloc[-horizon:]
    avec_meteo = meteo_utilisable(train, test, temperature)
    disponibles = [m for m in COULEURS_MODELES
                   if (m != "Prophet" or prophet_disponible()) and (m not in MODELES_METEO or avec_meteo)]
    choisis = c2.multiselect("Modèles", disponibles, default=[m for m in ("Naive Drift", "Naive saisonnier", "XGBoost")
                                                               if m in disponibles])
    if temperature is not None and not avec_meteo:
        st.caption("Les modèles avec température sont masqués : le fichier météo ne couvre pas toute la période "
                   "de test, ou laisse moins de deux ans d'entraînement.")
    elif avec_meteo:
        st.caption(f"Les modèles avec température s'entraînent à partir du {max(train.index[0], temperature.index[0]):%d/%m/%Y}, "
                   "date de début du fichier météo. La température de la période de test est la température observée : "
                   "leur avantage est donc un maximum, qu'une vraie prévision météo réduirait.")

    parametres = {m: {} for m in choisis}
    if "Naive saisonnier" in choisis:
        parametres["Naive saisonnier"]["k"] = st.select_slider(
            "Naive saisonnier : période répétée (jours)", options=[7, 14, 28], value=7)
    if "SARIMA" in choisis:
        with st.expander("Ordres du modèle SARIMA", expanded=True):
            auto = st.checkbox("Trouver les ordres avec auto_arima (période de 7 jours)", value=False)
            if auto:
                order, seasonal_order, trend = ordres_auto(train)
            else:
                cols = st.columns(6)
                order = tuple(cols[i].number_input(n, 0, 5, v) for i, (n, v) in enumerate(zip("pdq", (2, 1, 1))))
                seasonal_order = tuple(cols[3 + i].number_input(n, 0, 2, v) for i, (n, v) in enumerate(zip("PDQ", (1, 0, 1)))) + (7,)
                trend = None
            st.caption(f"Modèle utilisé : SARIMA{order}{seasonal_order}, constante : {'oui' if trend else 'non'}")
            parametres["SARIMA"] = {"order": order, "seasonal_order": seasonal_order, "trend": trend}

    if not choisis:
        st.info("Choisissez au moins un modèle.")
        return

    resultats = {m: prevoir(m, train, horizon, parametres[m], temperature if m in MODELES_METEO else None)
                 for m in choisis}
    previsions = {m: prevision for m, (prevision, _) in resultats.items()}
    if "VARIMA (conso + température)" in resultats:
        p, d = resultats["VARIMA (conso + température)"][1]
        st.caption(f"VARIMA retenu : p = {p} (AIC d'un VAR), d = {d} (test de Dickey-Fuller), q = 0.")

    fig = go.Figure()
    historique = journaliere.iloc[-(horizon + 365):]
    fig.add_scatter(x=historique.index, y=historique, name="Consommation réelle",
                    line=dict(color=COULEUR_REEL, width=1.5))
    for nom, prevision in previsions.items():
        fig.add_scatter(x=prevision.index, y=prevision, name=nom, line=dict(color=COULEURS_MODELES[nom], width=2))
    fig.add_vline(x=test.index[0], line=dict(color=COULEUR_REEL, dash="dot", width=1))
    fig.add_annotation(x=test.index[0], y=1, yref="paper", text="début du test", showarrow=False,
                       xanchor="left", font=dict(size=12))
    fig.update_yaxes(title=UNITE)
    st.plotly_chart(mise_en_forme(fig, 460))

    st.subheader("Comparaison sur la période de test")
    scores = pd.DataFrame(
        {
            nom: {
                "MAPE (%)": modeles.mape(test, prevision),
                "MAE (MW)": float(np.mean(np.abs(test.to_numpy() - prevision.to_numpy()))),
                "RMSE (MW)": float(np.sqrt(np.mean((test.to_numpy() - prevision.to_numpy()) ** 2))),
            }
            for nom, prevision in previsions.items()
        }
    ).T.sort_values("MAPE (%)")
    st.dataframe(scores.style.format("{:.2f}"))
    meilleur = scores.index[0]
    st.markdown(f"Meilleur modèle sur cette période : **{meilleur}** (MAPE de {scores.loc[meilleur, 'MAPE (%)']:.2f} %).")


# --- Page --------------------------------------------------------------------

st.title("⚡ Consommation électrique des Hauts-de-France")
st.caption("Analyse et prévision de séries temporelles")

contenu, nom = source_des_donnees()
if nom is None:
    st.warning(
        "Aucune donnée trouvée. Placez le fichier RTE dans `data/data.csv` (ou le fichier `data/hdf_daily.csv` "
        "produit par le notebook `3_model`), ou chargez un fichier CSV dans la barre latérale."
    )
    st.stop()

demi_horaire, journaliere = charger(contenu, nom, date_modification(contenu, nom))
contenu_meteo, nom_meteo = source_temperature()
temperature = None
if nom_meteo is not None:
    temperature = charger_temperature(contenu_meteo, nom_meteo, date_modification(contenu_meteo, nom_meteo))

titres = ["Présentation", "Exploration", "Saisonnalité et stationnarité"]
titres += (["Météo"] if temperature is not None else []) + ["Prévisions"]
onglets = dict(zip(titres, st.tabs(titres)))
with onglets["Présentation"]:
    onglet_presentation(journaliere)
with onglets["Exploration"]:
    onglet_exploration(demi_horaire, journaliere)
with onglets["Saisonnalité et stationnarité"]:
    onglet_stationnarite(journaliere)
if temperature is not None:
    with onglets["Météo"]:
        onglet_meteo(journaliere, temperature)
with onglets["Prévisions"]:
    onglet_previsions(journaliere, temperature)
