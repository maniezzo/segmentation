import warnings, time
import numpy as np
import pandas as pd
import sarimaModels as sm
import shortSeriesModels as ssm
from model_result import ModelResult

def read_series(idSeries):
   df = pd.read_csv("C:\git\segmentation\data\M3\M3month.csv")
   ds = pd.to_numeric(df.iloc[idSeries,6:].dropna(), errors="coerce")
   return df.iloc[idSeries,0],ds

if __name__ == '__main__':
   warnings.filterwarnings("ignore", category = UserWarning)
   name,data = read_series(0)
   m = 12
   lstModels = []

   tstart = time.perf_counter()

   t1 = 0
   for t1 in range (0,len(data)-8):
      for t2 in range(t1+8, len(data) + 1):
         print(f"tend - {t2}")
         series = data[t1:t2]
         if(len(series) <= m):
            res = ssm.linearInterpolation(series)
            lstModels.append((t1,t2,res.aic,res.rmse,res.model,res.seasonal_model,res.params))
            print(f"AR(1): {t1} - {t2},  {res.aic}")
         elif(len(series) <= 2*m):
            res = ssm.arimaAndLess(series,max_p=2,max_d=0)
            lstModels.append((t1,t2,res.aic,res.rmse,res.model,res.seasonal_model,res.params))
            print(f"AR(2): {t1} - {t2}, {res.aic}")
         elif(len(series) <= 5*m):
            res = ssm.arimaAndLess(series,max_p=2,max_d=1,max_q=1)
            lstModels.append((t1,t2,res.aic,res.rmse,res.model,(0,0,0,0),res.params))
            print(f"ARIMA: {t1} - {t2}, {res.aic}")
         else:
            fStatsModels = False
            if(fStatsModels):
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
   
            fStatsforecasts =True
            if(fStatsforecasts):
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
               lstModels.append((t1, t2, res.aic, res.rmse, res.model, res.seasonal_model, res.params))
   
   tcpu = time.perf_counter() - tstart
   print(f"tcpu {tcpu}")
   df = pd.DataFrame(lstModels)
   df.to_csv(f"data/{name}models.csv")
   print("fine: >>>>>>>>>>>>>>> LEGGI note.txt <<<<<<<<<<<<<<<<")