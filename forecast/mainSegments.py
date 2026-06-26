import warnings, time
import numpy as np
import pandas as pd
import sarimaModels as sm
import shortSeriesModels as ssm
from etstheta import go_HW,go_theta
from model_result import ModelResult
from statsmodels.tsa.seasonal import STL

def read_series(idSeries):
   df = pd.read_csv("C:\git\segmentation\data\M3\M3month.csv")
   ds = pd.to_numeric(df.iloc[idSeries,6:].dropna(), errors="coerce")
   return df.iloc[idSeries,0],ds

def forecast_cost(model,series,m,nforecast):
   if(model=="HW"):      ypred = go_HW(series[:-validation],m,nforecast)
   elif(model=="theta"): ypred = go_theta(series[:-validation],m,nforecast)
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

def sarima_cost(series):
   if (len(series) <= m):
      res = ssm.linearInterpolation(series)
      lstModels.append((t1, t2, res.aic, res.rmse, res.model, res.seasonal_model))
      print(f"AR(1): {t1} - {t2},  {res.aic}")
   elif (len(series) <= 2 * m):
      res = ssm.arimaAndLess(series, max_p=2, max_d=0)
      lstModels.append((t1, t2, res.aic, res.rmse, res.model, res.seasonal_model))
      print(f"AR(2): {t1} - {t2}, {res.aic}")
   elif (len(series) <= 5 * m):
      res = ssm.arimaAndLess(series, max_p=2, max_d=1, max_q=1)
      lstModels.append((t1, t2, res.aic, res.rmse, res.model, (0, 0, 0, 0)))
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

if __name__ == '__main__':
   warnings.filterwarnings("ignore", category = UserWarning)
   name,data = read_series(528)  # "N1680"
   m = 12
   minLength = 18
   lstModels = []
   model = "theta" # "theta" "HW" "SARIMA"
   # HW needs burn-in for internal initializations
   burnin = 0
   if (model == "HW"):
      burnin = m
   # in case of forecast costs
   validation = 0 # forecasted interval
   if(model == "HW" or model == "theta"):
      validation = 6

   minLength += burnin+validation
   tstart = time.perf_counter()
   
   # destagionalizzo
   stl = STL(data, period=m)
   res = stl.fit()
   seasonal = res.seasonal     # length n
   ydeseas  = data - seasonal  # for modelling
   # Last full seasonal cycle from STL
   last_cycle = seasonal[-m:]  # shape (m,)
   # For h-step-ahead forecast, the seasonal additive factor at step h is:
   seasonal_forecast = np.array([last_cycle[i % m] for i in range(validation)])
   
   t1 = 0
   for t1 in range (0,len(ydeseas)-minLength):
      for t2 in range(t1+minLength, len(ydeseas) + 1 - validation):
         print(f"t {t1} - {t2}")
         series = ydeseas[t1:t2]
         if(model == "HW" or model == "theta"):
            res = forecast_cost(model,series,m,validation)
         else:
            res = sarima_cost(series)
         lstModels.append((t1, t2, res.aic, res.rmse, res.model, res.seasonal_model))
   
   tcpu = time.perf_counter() - tstart
   print(f"tcpu {tcpu}")
   df = pd.DataFrame(lstModels)
   df.to_csv(f"data/{name}models.csv")
   print("fine: >>>>>>>>>>>>>>> LEGGI note.txt <<<<<<<<<<<<<<<<")