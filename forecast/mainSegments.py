import warnings, time, os
import numpy as np
import pandas as pd
import sarimaModels as sm
import shortSeriesModels as ssm
from models import go_HW,go_theta,go_RF,go_AR1
from model_result import ModelResult
#from statsmodels.tsa.seasonal import STL
#from util import deseason


def read_series(idSeries):
   df = pd.read_csv("C:\git\segmentation\data\M3\M3month.csv")
   ds = pd.to_numeric(df.iloc[idSeries,6:].dropna(), errors="coerce")
   return df.iloc[idSeries,0],ds

# modelli AR!, HW a ETS, costi come rmse previsioni
def forecast_cost(model, series, m, validation):
   if(model=="HW"):      _,ypred = go_HW(series[:-validation],validation,m=m) # m=0 se destaginalizzata
   elif(model=="theta"): _,ypred = go_theta(series[:-validation],validation,m=m)
   elif(model=="RF"):    _,ypred = go_RF(series[:-validation],validation,m=m) # m=0 se destaginalizzata
   elif(model=="AR1"):   _,ypred = go_AR1(series[:-validation],validation,m=m) # m=0 se destaginalizzata
   diff = series[-validation:] - ypred
   rmse = np.sqrt(np.dot(diff, diff) / ypred.size)
   res = ModelResult(
      aic=None,
      rmse=rmse,
      model=None,
      seasonal_model=None,
      params=None,
   )
   return res

def sarima_cost(series,lstResults,t1,t2,m):
   if (len(series) <= m):
      res = ssm.linearInterpolation(series)
      lstResults.append((t1, t2, res.aic, res.rmse, res.model, res.seasonal_model))
      print(f"AR(1): {t1} - {t2},  {res.aic}")
   elif (len(series) <= 2 * m):
      res = ssm.arimaAndLess(series, max_p=2, max_d=0)
      lstResults.append((t1, t2, res.aic, res.rmse, res.model, res.seasonal_model))
      print(f"AR(2): {t1} - {t2}, {res.aic}")
   elif (len(series) <= 5 * m):
      res = ssm.arimaAndLess(series, max_p=2, max_d=1, max_q=1)
      lstResults.append((t1, t2, res.aic, res.rmse, res.model, (0, 0, 0, 0)))
      print(f"ARIMA: {t1} - {t2}, {res.aic}")
   else:
      fStatsModels = False
      if (fStatsModels):
         print("--------------------------- statsmodels")
         tstart = time.perf_counter()
         for iter in range(100):
            res = sm.statsmodels_grid_search(
               y=series,
               m=12,  # seasonality
               criterion="aic"
            )
         tcpu = time.perf_counter() - tstart
         print(f"tcpu {tcpu}")
         print(res.aic)

         # validazione aic
         out = sm.sarima_aic_at_params(
            y=series,
            order=res.model,
            seasonal_order=res.seasonal_model,
            coef=res.params,
            sigma2=res.params['sigma2'],
         )

         print(f'AIC: {out["aic"]}')
         print(f'Log likelihood: {out["loglik"]}')

      fStatsforecasts = True
      if (fStatsforecasts):
         print("--------------------------- statsforecasts")
         df = pd.DataFrame({
            "unique_id": 1,  # series identifier (can be any constant if you have one series)
            "ds": pd.date_range(start="2000-01-01", periods=len(series), freq="ME"),  # datetime index
            "y": series.values,  # your values
         })

         sf = sm.statsforecasts_autoarima(m=12, n_jobs=-1, ic="aic")
         sf.fit(df)
         fitted_model = sf.fitted_[0][0]

         res = sm.extract_statsforecast_arima_info(fitted_model)
         print(f"SARIMA: {t1} - {t2}, {res.aic}, {res.rmse}")

         # validazione aic
         sf_coef = fitted_model.model_["coef"]
         sigma2 = fitted_model.model_["sigma2"]
         coef_sm = sm.statsforecast_to_statsmodels_coef(sf_coef, m=12)

         out = sm.sarima_aic_at_params(
            y=series,
            order=res.model,
            seasonal_order=res.seasonal_model,
            coef=coef_sm,
            sigma2=fitted_model.model_["sigma2"],
         )

         print(f'Log likelihood: {out["loglik"]}')
         print(f'AIC check: {out["aic"]}')
   return res
'''
deseasoning pipeline (v. aaatobackup/tematiche/forecast/data):
1) Estimate trend and seasonality jointly with STL using robust=True.
2) Compute Hyndman's seasonality strength Fs
3) Remove the seasonal component only if Fs >0.6 (or another threshold validated on your application).
4) Store FS as a metadata feature—it may prove useful later when analyzing the performance of forecasting methods on the deseasonalized series.

Hyndman: Fs = max(0, 1- var(R)/var(S+R)
S: seasonal component, R: ramainder
'''

# calcolo indice di stagionalità (hyndman)
def stl_seasonality_strength(x, period=12, robust=True):
    """
    Compute STL-based seasonality strength:
        Fs = max(0, 1 - Var(R) / Var(S + R))

    Returns:
        Fs, trend, seasonal, resid
    """
    x = np.asarray(x, dtype=float)

    # If missing values interpolate before this step

    if len(x) < 2 * period:
        return 0.0, None, None, None

    stl = STL(x,
              period=period,
              seasonal=len(x)*2-1,  # Large window forces it to look at all points
              seasonal_deg=0,   # Forces the seasonal component to be strictly periodic
              robust=robust)
    res = stl.fit()

    seasonal = res.seasonal
    trend = res.trend
    resid = res.resid

    denom = np.var(seasonal + resid, ddof=1)

    if denom <= 0:
        Fs = 0.0
    else:
        Fs = max(0.0, 1.0 - np.var(resid, ddof=1) / denom)

    return Fs, trend, seasonal, resid

# destagionalizzazione (se Fs>threshold)
def deseasonalize_if_needed(x, period=12, threshold=0.6):
    """
    Deseasonalize only if STL seasonality strength exceeds threshold.

    Returns a dictionary containing:
        original
        deseasonalized
        seasonality_strength
        removed_seasonality
        trend
        seasonal
        resid
    """
    x = np.asarray(x, dtype=float)

    Fs, trend, seasonal, resid = stl_seasonality_strength(x, period=period, robust=False)

    if seasonal is None:
        return {
            "deseasonalized": x,
            "seasonality_strength": Fs,
            "removed_seasonality": False,
            "seasonal": None,
        }

    if Fs > threshold:
        y = x - seasonal
        removed = True
    else:
        y = x.copy()
        removed = False

    return {
        "deseasonalized": y,
        "seasonality_strength": Fs,
        "removed_seasonality": removed,
        "seasonal": seasonal[0:period],
    }

# main della segmentazione
def go_segment(name, dfpoints, m, validation, minLength = 18, burnin = 0):
   if os.path.isfile(f'data/{name}models_{validation}.csv'):
      print("Segment file already exists. ")
   else:
      print("Segment file does not already exist. ")
      
      fDeseason = False
      if(fDeseason):
         # ----------------------------------------------------- destagionalizzo
         res     = deseasonalize_if_needed(dfpoints, period=m, threshold=0)  # threshold 0 destagionalizzo sempre, alto mai
         ydeseas = res['deseasonalized']
         yseries = ydeseas
         coeff_seas = res['seasonal']  # first full seasonal cycle from STL
      else:
         yseries = np.array(dfpoints)
      
      lstModels = ["AR1", "HW", "theta", "RF"]
      tstart = time.perf_counter()
      lstResults = []
      # HW would need burn-in for internal initializations
      # if (model == "HW"):
      #   burnin = m
      minLength += burnin + validation
      
      for idModel in range(len(lstModels)):
         model = lstModels[idModel]
   
         cont = 0
         for t1 in range (0,len(yseries)-minLength):
            for t2 in range(t1+minLength, len(yseries) + 1 - validation):
               print(f"{model}) t {t1} - {t2}")
               if(idModel==0):
                  lstResults.append([t1,t2,np.nan,np.nan,np.nan,np.nan])
               if(lstResults[cont][0] != t1 or lstResults[cont][1] != t2):
                  input("Press Enter to continue...")
                  
               series = yseries[t1:t2]
               if(model in lstModels):  # sempre, ma giusto per tenere la chiamata dopo
                  res = forecast_cost(model,series,m,validation)
               else:
                  res = sarima_cost(series)
               lstResults[cont][idModel+2] = res.rmse
               cont += 1
      
      tcpu = time.perf_counter() - tstart
      print(f"tcpu segmentation {tcpu}")
      df = pd.DataFrame(lstResults)
      df.columns = ["t1", "t2", "AR1", "HW", "theta", "RF"]
      df.to_csv(f"data/{name}models_{validation}.csv")
      #df.to_pickle(f"data/{name}models_{validation}.pkl") # per file binari, si guadagna poco

if __name__ == '__main__':
   warnings.filterwarnings("ignore", category = UserWarning)
   name,dfpoints = read_series(528)  # "N1680"

   model = "theta" # "theta" "HW" "SARIMA"
   go_segment(name,dfpoints,model)
   print("fine")