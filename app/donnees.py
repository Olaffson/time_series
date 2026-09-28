"""Chargement et préparation de la consommation électrique des Hauts-de-France."""

from pathlib import Path

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

DOSSIER_DATA = Path(__file__).resolve().parent.parent / "data"
CODE_HAUTS_DE_FRANCE = 32
COLONNE = "consommation_brute_electricite_rte"


def lire_brut(source):
    """Lit le CSV RTE (séparateur `;`) ou le fichier journalier `hdf_daily.csv` produit par le notebook."""
    if hasattr(source, "read"):
        entete = source.readline()
        entete = entete.decode("utf-8") if isinstance(entete, bytes) else entete
        source.seek(0)
    else:
        with open(source, encoding="utf-8") as f:
            entete = f.readline()
    entete = entete.lstrip("\ufeff")
    sep = ";" if entete.count(";") > entete.count(",") else ","
    colonnes = [c.strip().strip('"') for c in entete.split(sep)]
    usecols = [c for c in ("date_heure", "code_insee_region", COLONNE) if c in colonnes]
    return pd.read_csv(source, sep=sep, usecols=usecols)


def serie_demi_horaire(df):
    """Série au pas de 30 minutes des Hauts-de-France, triée et en heure de Paris."""
    if "code_insee_region" in df.columns:
        df = df[df["code_insee_region"] == CODE_HAUTS_DE_FRANCE]
    serie = df.set_index(pd.to_datetime(df["date_heure"], utc=True))[COLONNE]
    serie = serie.sort_index().dropna()
    serie.index = serie.index.tz_convert("Europe/Paris")
    return serie


def serie_journaliere(serie):
    """Moyenne journalière, sans fuseau horaire, avec les jours manquants interpolés."""
    journaliere = serie.resample("D").mean()
    journaliere.index = journaliere.index.tz_localize(None)
    return journaliere.asfreq("D").interpolate()


def desaisonnaliser(journaliere, periode=365):
    """Retire la composante saisonnière annuelle (décomposition additive)."""
    saisonnalite = seasonal_decompose(journaliere, model="additive", period=periode).seasonal
    return journaliere - saisonnalite, saisonnalite


FICHIER_TEMPERATURE = "temperature-quotidienne-regionale.csv"


def lire_temperature(source):
    """Température moyenne quotidienne des Hauts-de-France (fichier ODRE, séparateur `;`)."""
    df = pd.read_csv(source, sep=";", usecols=["Date", "Code INSEE région", "TMoy (°C)"])
    df = df[df["Code INSEE région"] == CODE_HAUTS_DE_FRANCE]
    temperature = df.set_index(pd.to_datetime(df["Date"]))["TMoy (°C)"].sort_index()
    temperature = temperature[~temperature.index.duplicated()]
    temperature.index.name = None
    temperature.name = "temperature"
    return temperature.asfreq("D").interpolate()
