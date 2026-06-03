import warnings, time
import numpy as np
import pandas as pd
import sarimaModels as sm
import shortSeriesModels as ssm
from model_result import ModelResult

def read_series():
   df = pd.read_csv("C:\git\segmentation\data\M3\M3month.csv")
   ds = pd.to_numeric(df.iloc[0,6:].dropna(), errors="coerce")
   return ds

if __name__ == '__main__':
   warnings.filterwarnings("ignore", category = UserWarning)
   data = read_series()
   m = 12

   tstart = time.perf_counter()

   for tend in range(8,len(data)+1):
      print(f"tend - {tend}")
      series = data[0:tend]
      if(len(series) <= m):
         res = ssm.linearInterpolation(series)
         print(f"AR(1): {0} - {tend},  {res.aic}")
      elif(len(series) <= 2*m):
         res = ssm.arimaAndLess(series,max_p=2,max_d=0)
         print(f"AR(2): {0} - {tend}, {res.aic}")
      elif(len(series) <= 5*m):
         res = ssm.arimaAndLess(series,max_p=2,max_d=1,max_q=1)
         print(f"ARIMA: {0} - {tend}, {res.aic}")
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
            print(f"SARIMA: {0} - {tend}, {res.aic}")

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

            print(f'AIC: {out["aic"]}')
            print(f'Log likelihood: {out["loglik"]}')
            print(f'AIC check: {out["aic"]}')

   tcpu = time.perf_counter() - tstart
   print(f"tcpu {tcpu}")
   print("fine: >>>>>>>>>>>>>>> LEGGI note.txt <<<<<<<<<<<<<<<<")