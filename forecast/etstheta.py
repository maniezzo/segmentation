import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.datasets import get_rdataset
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ETS, Holt Winters
def go_HW(y,m,nforecast):
   model = ExponentialSmoothing(y, seasonal_periods=m,trend="add",
       seasonal     ="mul",
       damped_trend = True,
       use_boxcox   = True,
       initialization_method = "estimated")
   hwfit = model.fit()
   # make forecast
   yfore = hwfit.predict(len(y), len(y)+nforecast-1)
   return yfore.values

# Theta
def go_theta(y,m,nforecast):
   # Fit the Theta model
   theta_model = ThetaModel(y,period=m)
   fit = theta_model.fit()
   #print(fit.summary())
   yfore = fit.forecast(steps=nforecast)  # Forecast nforecast months ahead
   return yfore.values

if __name__ == '__main__':
   airpass = get_rdataset('AirPassengers').data.value.values
   
   period = 12
   # STL decomposition
   stl = STL(airpass, period=period, robust=True).fit()
   # Deseasonalize
   deseasoned = airpass - stl.seasonal
   # Forecast deseasonalized (naive)
   forecast_deseasoned = np.mean(deseasoned[-12:]) * np.ones(period)
   # Extend seasonal component
   last_seasonal_cycle = stl.seasonal[-period:]
   forecast_seasonal = np.tile(last_seasonal_cycle, (len(forecast_deseasoned) // period + 1))[:len(forecast_deseasoned)]
   
   ytheta = go_theta(deseasoned)
   yHW = go_HW(deseasoned)

   # Reseasonalize
   ySTL = forecast_deseasoned + forecast_seasonal
   ytheta += forecast_seasonal
   yHW += forecast_seasonal

   # Plot the original series and forecast
   plt.figure(figsize=(10, 5))
   plt.plot(airpass, label='Actual', color='blue')
   plt.plot(range(len(airpass), len(airpass) + 12), ytheta, label='Theta Forecast')
   plt.plot(range(len(airpass), len(airpass) + 12), yHW, label='Holt-Winters Forecast')
   plt.plot(range(len(airpass), len(airpass) + 12), ySTL, label='STL Forecast')
   plt.xlabel('Year')
   plt.ylabel('Passengers')
   plt.title('Theta / HW Forecast for Airline Passengers')
   plt.legend()
   plt.show()
