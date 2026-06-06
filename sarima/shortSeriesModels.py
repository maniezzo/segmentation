import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from model_result import ModelResult

# interpolazione lineare dei punti
def linearInterpolation(y):
   y = np.asarray(y, dtype=float)
   x = np.arange(len(y))

   slope, intercept = np.polyfit(x, y, deg=1)
   yhat = intercept + slope * x
   rss = np.sum((y - yhat) ** 2)
   
   rmse = np.sqrt(np.mean((y - yhat) ** 2))
   
   n = len(y)
   k = 3  # intercept, slope, sigma^2
   aic = n * np.log(rss / n) + 2 * k + n * (1 + np.log(2 * np.pi))
   res = ModelResult(
      aic=aic,
      rmse=rmse,
      model=(1, 0, 0),
      seasonal_model=(0, 0, 0, 0),
      params=(slope, intercept),
   )
   return res

# AR, ARMA e ARIMA tutto con ARIMA di statsmodels
def arimaAndLess(y, max_p=2, max_d=1, max_q=0, criterion="aic"):
   best = None
   rows = []

   for d in range(max_d + 1):
      for p in range(max_p + 1):
         for q in range(max_q + 1):
            order = (p, d, 0)

            try:
               res = ARIMA(
                 y,
                 order=order,
                 trend=None if d > 0 else "c"
               ).fit()
               rmse = np.sqrt(np.mean(res.resid ** 2))
               
               rows.append({
                 "order": order,
                 "seasonal_model": (0, 0, 0, 0),
                 "aic": res.aic,
                 "bic": res.bic,
                 "rmse": rmse,
                 "params": res.params.to_dict()
               })

               if best is None or getattr(res, criterion) < getattr(best, criterion):
                  best = ModelResult(
                     aic=res.aic,
                     rmse=rmse,
                     model=order,
                     seasonal_model=(0, 0, 0, 0),
                     params=res.params.to_dict(),
                  )
            except Exception as e:
               rows.append({
                    "order": order,
                    "seasonal_model": (0, 0, 0, 0),
                    "aic": np.nan,
                    "bic": np.nan,
                    "rmse": np.nan,
                    "error": str(e)
                  })

   #table = pd.DataFrame(rows).sort_values(criterion, na_position="last")
   return best