import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.datasets import get_rdataset
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor

# random forest
def go_RF(y,m,nforecast):
   # --- 1. Setup & Train/Validation Split ---
   lags = 12  # Number of past observations to use as input features
   
   # --- 2. Create Lagged Features (Supervised Matrix) ---
   X, Y = [], []
   for i in range(lags, len(y)):
      X.append(y[i - lags:i])
      Y.append(y[i])
   
   X = np.array(X)
   Y = np.array(Y)
   
   # --- 3. Train/Validation Split ---
   # Keep the last m values for validation
   X_train, Y_train = X[:-m], Y[:-m]
   X_val, Y_val = X[-m:], Y[-m:]
   
   # --- 4. Train Random Forest ---
   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train, Y_train)
   
   # --- 5. Forecast (Direct Validation Predictions) ---
   # Predicting the validation set using actual historical lag features
   yfore = model.predict(X_val)
   
   # Optional: Evaluate accuracy
   mae = np.mean(np.abs(yfore - Y_val))
   print(f"Validation MAE: {mae:.4f}")
   return model,yfore

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
   theta_model = ThetaModel(y,period=m)
   fit = theta_model.fit()
   #print(fit.summary())
   yfore = fit.forecast(steps=nforecast)  # Forecast nforecast points ahead
   return yfore.values

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
   forecast_deseasoned = np.mean(deseasoned[-12:]) * np.ones(m)
   # Extend seasonal component
   last_seasonal_cycle = stl.seasonal[-m:]
   forecast_seasonal = np.tile(last_seasonal_cycle, (len(forecast_deseasoned) // m + 1))[:len(forecast_deseasoned)]
   
   ytheta = go_theta(deseasoned,m,nforecast)
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
   plt.plot(range(len(y) - 12, len(y)), ytheta, label='Theta Forecast')
   plt.plot(range(len(y) - 12, len(y)), yHW, label='Holt-Winters Forecast')
   plt.plot(range(len(y) - 12, len(y)), yRF, label='Random forest Forecast')
   plt.plot(range(len(y) - 12, len(y)), ySTL, label='STL Forecast')
   plt.xlabel('Year')
   plt.ylabel('y')
   plt.title('Theta / HW / RF Forecast')
   plt.legend()
   plt.show()
