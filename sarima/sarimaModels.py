import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
from model_result import ModelResult

# statsmodels'
def statsmodels_grid_search(
        y,
        m,
        criterion="aic",
        max_p=2,
        max_q=2,
        max_d=1,
        max_D=1,
        trend=None,
        enforce_stationarity=False,
        enforce_invertibility=False,
        maxiter=50,
):
   """
   Restricted SARIMA selection with:
     p,q <= 2
     d,D <= 1
     P = Q = 0
     known seasonality m

   Returns:
     best_result, results_table
   """

   y = pd.Series(y).astype(float).dropna()

   best_res = None
   rows = []

   for d in range(max_d + 1):
      for D in range(max_D + 1):
         for p in range(max_p + 1):
            for q in range(max_q + 1):

               # Optional: skip completely empty ARMA part if both differencing are zero
               # if p == q == d == D == 0:
               #     continue

               order = (p, d, q)
               seasonal_order = (0, D, 0, m)

               try:
                  with warnings.catch_warnings():
                     warnings.simplefilter("ignore")

                     model = SARIMAX(
                        y,
                        order=order,
                        seasonal_order=seasonal_order,
                        trend=trend,
                        enforce_stationarity=enforce_stationarity,
                        enforce_invertibility=enforce_invertibility,
                     )

                     res = model.fit(disp=False, maxiter=maxiter)

                  rows.append({
                     "order": order,
                     "seasonal_order": seasonal_order,
                     "aic": res.aic,
                     "bic": res.bic,
                     "hqic": res.hqic,
                     "llf": res.llf,
                     "converged": res.mle_retvals.get("converged", None),
                  })

                  score = getattr(res, criterion)

                  if np.isfinite(score):
                     if best_res is None or score < getattr(best_res, criterion):
                        best_res = res

               except Exception as e:
                  rows.append({
                     "order": order,
                     "seasonal_order": seasonal_order,
                     "aic": np.nan,
                     "bic": np.nan,
                     "hqic": np.nan,
                     "llf": np.nan,
                     "converged": False,
                     "error": str(e),
                  })

   # qui tutti i modelli testati
   # table = pd.DataFrame(rows).sort_values(criterion, na_position="last")
   res = ModelResult(
            aic=best_res.aic,
            model=best_res.model.order,
            seasonal_model=best_res.model.seasonal_order,
            params=dict(zip(best_res.param_names, best_res.params)),
            fitted=best_res
         )

   return res


# statsforecast's
def statsforecasts_autoarima(m, n_jobs=-1, ic="aic"):
   model = AutoARIMA(
      season_length=m,
      max_p=2,
      max_q=2,
      max_d=1,
      max_D=1,
      max_P=0,
      max_Q=0,
      start_p=0,
      start_q=0,
      start_P=0,
      start_Q=0,
      seasonal=True,
      stepwise=True,  # True
      ic=ic,  # bic
      approximation=True, # True
      allowmean=True, # True
      allowdrift = True, # True
      trace=False,
   )

   res = StatsForecast(
      models=[model],
      freq="ME",  # "ME" monthly, "Q" quaterly, ...
      n_jobs=n_jobs
   )
   return res

# statsforecast, recupera ordini e parametri
def extract_statsforecast_arima_info(fitted_model):
   p, q, P, Q, m_fit, d, D = fitted_model.model_["arma"]
   seasonal_used = (P > 0) or (Q > 0) or (D > 0)
   if not seasonal_used:
      seasonal_order = (0, 0, 0, 0)
   else:
      seasonal_order = (P, D, Q, m_fit)
   print(fitted_model.model_.keys())
   res = ModelResult(
           aic=fitted_model.model_["aic"],
           model=(p, d, q),
           seasonal_model=seasonal_order,
           params=fitted_model.model_.get("coef", {}),
           fitted=fitted_model,
       )
   return res

# ricalcolo AIC via filtro di kalman, usa i parametri di statsmodels
def sarima_aic_at_params(
    y,
    order,
    seasonal_order,
    coef,
    sigma2=None,
    trend=None,
):
    y = pd.Series(y).astype(float).dropna()

    mod = SARIMAX(
        y,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    params = []
    names = mod.param_names

    for name in names:
        if name == "sigma2":
            if sigma2 is None:
                raise ValueError("sigma2 is required")
            params.append(sigma2)
        else:
            params.append(coef[name])

    res = mod.filter(params) # qui non rifitta, usa kalman

    return {
        "aic": res.aic,
        "loglik": res.llf,
        "k_params": len(params),
        "param_names": names,
        "params": params,
    }

# cambia i nomi dei parametri di statsforecast così posso chiamare la funzione sopra
def statsforecast_to_statsmodels_coef(sf_coef, m):
   out = {}

   for k, v in sf_coef.items():
      if k.startswith("ar"):
         out[f"ar.L{k[2:]}"] = v
      elif k.startswith("ma"):
         out[f"ma.L{k[2:]}"] = v
      elif k.startswith("sar"):
         lag = int(k[3:]) * m
         out[f"ar.S.L{lag}"] = v
      elif k.startswith("sma"):
         lag = int(k[3:]) * m
         out[f"ma.S.L{lag}"] = v
      elif k in ("intercept", "const"):
         out["intercept"] = v
      else:
         out[k] = v

   return out