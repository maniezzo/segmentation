import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

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
   
   table = pd.DataFrame(rows).sort_values(criterion, na_position="last")
   
   return best_res, table

# statsforecast's
def statsforecasts_autoarima(m, n_jobs=-1, ic="aic"):
   warnings.filterwarnings("ignore", message="possible convergence problem")
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
         stepwise=False, # True
         ic=ic, # bic
         approximation=True,
         trace=False,
      )
   
   res = StatsForecast(
            models=[model],
            freq="ME",      #  "ME" monthly, "Q" quaterly, ...
            n_jobs=n_jobs
         )
   return res


def extract_statsforecast_arima_info(fitted_model):
   p, q, P, Q, m, d, D = fitted_model.model_["arma"]

   return {
      "order": (p, d, q),
      "seasonal_order": (P, D, Q, m),
      "coef": fitted_model.model_.get("coef", {}),
      "sigma2": fitted_model.model_.get("sigma2"),
      "aic": fitted_model.model_.get("aic"),
      "bic": fitted_model.model_.get("bic"),
      "aicc": fitted_model.model_.get("aicc"),
      "loglik": fitted_model.model_.get("loglik"),
   }
def read_series():
   df = pd.read_csv("C:\git\segmentation\data\M3\M3month.csv")
   ds = pd.to_numeric(df.iloc[0,6:].dropna(), errors="coerce")
   return ds

if __name__ == '__main__':
   series = read_series()
   
   print("--------------------------- statsmodels")
   best_model, table = statsmodels_grid_search(
      y=series,
      m=12,  # seasonality
      criterion="aic"
   )
   
   print(table.head())
   print(best_model.summary())

   from statsforecast.utils import AirPassengersDF

   # modello su tutte le colonne di un dataframe, n_jobs dice quali
   print("--------------------------- statsforecasts")
   df = pd.DataFrame({
      "unique_id": 1,  # series identifier (can be any constant if you have one series)
      "ds": pd.date_range(start="2000-01-01", periods=len(series), freq="ME"),  # datetime index
      "y": series.values,  # your values
   })
   sf = statsforecasts_autoarima(m=12, n_jobs=-1, ic="aic")
   sf.fit(df)
   fitted_model = sf.fitted_[0][0]

   info = extract_statsforecast_arima_info(fitted_model)
   print(f"Order: {info['order']}")
   print(f"Seasonal order: {info['seasonal_order']}")
   print(f"coef: {info['coef']}")
   # forecast = sf.forecast(df=df,h=14) # ha dentro il fit
   
   print("fine")