import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.datasets import get_rdataset
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.ar_model import AutoReg

# AR(1)
def go_AR1(y, m, nforecast):
    # m is unused, kept only for compatibility
    model = AutoReg(y, lags=1, old_names=False)
    arfit = model.fit()

    # make forecast
    yfore = arfit.predict(
        start=len(y),
        end=len(y) + nforecast - 1
    )

    return arfit, yfore

# random forest
def go_RF(y, m, nforecast):
   """
   Train a Random Forest on the complete input series and recursively
   forecast the next nforecast out-of-sample observations.

   Parameters
   ----------
   y : array-like
       Historical time series.
   m : int
       Retained for compatibility. Currently unused.
   nforecast : int
       Number of future observations to forecast.

   Returns
   -------
   model : RandomForestRegressor
       Fitted Random Forest model.
   yfore : np.ndarray
       Recursive out-of-sample forecasts.
   """
   y = np.asarray(y, dtype=float).reshape(-1)
   # Choose the number of lagged observations.
   #lags = 12 if len(y) < 24 else 18
   lags = 12
   
   if len(y) <= lags:
      raise ValueError(
         f"The series must contain more than {lags} observations."
      )
   
   # Create the training matrix
   X, Y = [], []
   for i in range(lags, len(y)):
      X.append(y[i - lags:i])
      Y.append(y[i])
   
   X = np.asarray(X)
   Y = np.asarray(Y)
   
   # Train on all available observations.
   model = RandomForestRegressor(
      n_estimators=100,
      random_state=42
   )
   model.fit(X, Y)
   
   # Start with the most recent lag window.
   window = y[-lags:].copy()
   forecasts = []
   
   # Recursive sliding-window forecasting.
   for _ in range(nforecast):
      next_forecast = model.predict(window.reshape(1, -1))[0]
      forecasts.append(next_forecast)
      
      # Remove the oldest observation and append the forecast.
      window = np.append(window[1:], next_forecast)
   
   yfore = np.asarray(forecasts)
   return model, yfore

# ETS, Holt Winters
def go_HW(y,m,nforecast):
   # no seasonality (Double Exponential Smoothing)
   model = ExponentialSmoothing(y,
       trend    = "add",
       seasonal = None,
       initialization_method = "estimated")
   hwfit = model.fit()
   # make forecast
   yfore = hwfit.predict(len(y), len(y)+nforecast-1)
   return hwfit,yfore

# Theta
def go_theta(y,m,nforecast):
   # Fit the Theta model
   theta_model = ThetaModel(y, deseasonalize=False) # lavoro su dati destagionalizzati
   fit = theta_model.fit()
   #print(fit.summary())
   yfore = fit.forecast(steps=nforecast)  # Forecast nforecast points ahead
   return fit,yfore.values

if __name__ == '__main__':
   #y = get_rdataset('AirPassengers').data.value.values
   y = pd.read_csv('../data/M3/N1906.csv',usecols=[1]).values.flatten()
   m = 12
   nforecast = 12
   # STL decomposition
   stl = STL(y, period=m, robust=True).fit()
   # Deseasonalize
   deseasoned = y - stl.seasonal
   # Forecast deseasonalized (naive)
   forecast_deseasoned = np.mean(deseasoned[-nforecast:]) * np.ones(m)
   # Extend seasonal component
   last_seasonal_cycle = stl.seasonal[-m:]
   forecast_seasonal = np.tile(last_seasonal_cycle, (len(forecast_deseasoned) // m + 1))[:len(forecast_deseasoned)]
   
   _,ytheta  = go_theta(deseasoned,m,nforecast)
   hwfit,yHW = go_HW(deseasoned,m,nforecast)
   rffit,yRF = go_RF(deseasoned,m,nforecast)

   # Reseasonalize
   ySTL = forecast_deseasoned + forecast_seasonal
   ytheta += forecast_seasonal
   yHW += forecast_seasonal
   yRF += forecast_seasonal

   # Plot the original series and forecast
   plt.figure(figsize=(10, 5))
   plt.plot(y, label='Actual', color='blue')
   plt.plot(range(len(y) - nforecast, len(y)), ytheta, label='Theta Forecast')
   plt.plot(range(len(y) - nforecast, len(y)), yHW, label='Holt-Winters Forecast')
   plt.plot(range(len(y) - nforecast, len(y)), yRF, label='Random forest Forecast')
   plt.plot(range(len(y) - nforecast, len(y)), ySTL, label='STL Forecast')
   plt.xlabel('Year')
   plt.ylabel('y')
   plt.title('Theta / HW / RF Forecast')
   plt.legend()
   plt.show()
