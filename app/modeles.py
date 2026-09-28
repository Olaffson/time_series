"""Modèles de prévision de la consommation journalière.

Chaque fonction s'entraîne sur `train` et prévoit `horizon` jours. Toutes les
prévisions sont exprimées en consommation réelle (MW) pour que les MAPE soient
comparables : les modèles qui travaillent sur la série désaisonnalisée
réintègrent ensuite la saisonnalité annuelle estimée sur l'entraînement.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

PERIODE_ANNUELLE = 365


def mape(reel, prevu):
    reel, prevu = np.asarray(reel), np.asarray(prevu)
    return float(np.mean(np.abs((reel - prevu) / reel)) * 100)


def _index_futur(train, horizon):
    return pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")


def _decomposer(train):
    """Série désaisonnalisée et saisonnalité annuelle prolongée, estimées sur l'entraînement seul."""
    saisonnalite = seasonal_decompose(train, model="additive", period=PERIODE_ANNUELLE).seasonal
    n = len(train)

    def prolonger(horizon):
        # seasonal_decompose répète le même motif tous les 365 points
        return saisonnalite.iloc[[(n + h) % PERIODE_ANNUELLE for h in range(horizon)]].to_numpy()

    return train - saisonnalite, prolonger


def naive_drift(train, horizon):
    from darts import TimeSeries
    from darts.models import NaiveDrift

    desaison, saisonnalite_future = _decomposer(train)
    modele = NaiveDrift().fit(TimeSeries.from_series(desaison, freq="D"))
    prevision = modele.predict(horizon).values().ravel()
    return pd.Series(prevision + saisonnalite_future(horizon), index=_index_futur(train, horizon))


def naive_saisonnier(train, horizon, k=7):
    from darts import TimeSeries
    from darts.models import NaiveSeasonal

    desaison, saisonnalite_future = _decomposer(train)
    modele = NaiveSeasonal(K=k).fit(TimeSeries.from_series(desaison, freq="D"))
    prevision = modele.predict(horizon).values().ravel()
    return pd.Series(prevision + saisonnalite_future(horizon), index=_index_futur(train, horizon))


def sarima(train, horizon, order, seasonal_order, trend=None):
    desaison, saisonnalite_future = _decomposer(train)
    resultats = sm.tsa.SARIMAX(desaison, order=order, seasonal_order=seasonal_order, trend=trend).fit(disp=False)
    prevision = resultats.get_forecast(horizon).predicted_mean.to_numpy()
    return pd.Series(prevision + saisonnalite_future(horizon), index=_index_futur(train, horizon))


def ordres_auto_arima(train):
    """Ordres SARIMA (m=7) et constante trouvés par auto_arima sur la série désaisonnalisée."""
    from pmdarima import auto_arima

    desaison, _ = _decomposer(train)
    modele = auto_arima(desaison, seasonal=True, m=7, error_action="ignore", suppress_warnings=True)
    return modele.order, modele.seasonal_order, ("c" if modele.with_intercept else None)


def _variables_calendaires(index):
    return pd.DataFrame(
        {
            "dayofweek": index.dayofweek,
            "dayofyear": index.dayofyear,
            "month": index.month,
            "quarter": index.quarter,
            "year": index.year,
        },
        index=index,
    )


def xgboost(train, horizon, temperature=None):
    """XGBoost sur des variables calendaires, et sur la température observée si elle est fournie."""
    from xgboost import XGBRegressor

    index = _index_futur(train, horizon)
    x_train, x_futur = _variables_calendaires(train.index), _variables_calendaires(index)
    if temperature is not None:
        x_train["temperature"] = temperature.reindex(train.index).to_numpy()
        x_futur["temperature"] = temperature.reindex(index).to_numpy()
    modele = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
    modele.fit(x_train, train)
    return pd.Series(modele.predict(x_futur), index=index)


def varima(train, horizon, temperature, max_lags=14):
    """VARIMA(p, d, 0) sur la consommation et la température désaisonnalisées.

    d vient du test de Dickey-Fuller, p de l'AIC d'un VAR. Renvoie la prévision de consommation
    (saisonnalité annuelle réintégrée) et les ordres (p, d).
    """
    import warnings

    from darts import TimeSeries
    from darts.models import VARIMA
    from statsmodels.tsa.stattools import adfuller

    conso_desaison, saisonnalite_future = _decomposer(train)
    temperature_desaison, _ = _decomposer(temperature.reindex(train.index))
    series = pd.DataFrame({"consommation": conso_desaison, "temperature": temperature_desaison})

    d = 1 if max(adfuller(series[c])[1] for c in series.columns) > 0.05 else 0
    selection = series.diff().dropna() if d == 1 else series
    p = max(1, sm.tsa.VAR(selection).select_order(maxlags=max_lags).aic)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modele = VARIMA(p=p, d=d, q=0).fit(TimeSeries.from_dataframe(series, freq="D"))
        prevision = modele.predict(horizon).to_dataframe()["consommation"].to_numpy()
    return pd.Series(prevision + saisonnalite_future(horizon), index=_index_futur(train, horizon)), (p, d)


def prophet(train, horizon):
    from prophet import Prophet

    modele = Prophet()
    modele.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))
    futur = pd.DataFrame({"ds": _index_futur(train, horizon)})
    return pd.Series(modele.predict(futur)["yhat"].to_numpy(), index=futur["ds"])
