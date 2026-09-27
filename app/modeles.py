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


def xgboost(train, horizon):
    from xgboost import XGBRegressor

    modele = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=0)
    modele.fit(_variables_calendaires(train.index), train)
    index = _index_futur(train, horizon)
    return pd.Series(modele.predict(_variables_calendaires(index)), index=index)


def prophet(train, horizon):
    from prophet import Prophet

    modele = Prophet()
    modele.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}))
    futur = pd.DataFrame({"ds": _index_futur(train, horizon)})
    return pd.Series(modele.predict(futur)["yhat"].to_numpy(), index=futur["ds"])
