import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.theta import ThetaModel
import util

# Theta
def go_theta(y,m,nforecast):
   # Fit the Theta model
   theta_model = ThetaModel(y,period=m)
   fit = theta_model.fit()
   #print(fit.summary())
   yfore = fit.forecast(steps=nforecast)  # Forecast nforecast points ahead
   return yfore.values


if __name__ == "__main__":
   name="N1930" #"N1679" "N1402"
   nameCheck,dfpoints = util.read_series(name)
   df = pd.read_csv("data/"+name+"models.csv",usecols=['0','1','2','3','4'])
   y = dfpoints.values
   
   stl = STL(y,
             period=12,
             seasonal=len(y) * 2 - 1,  # Large window forces it to look at all points
             seasonal_deg=0,  # Forces the seasonal component to be strictly periodic
             robust=True)
   res = stl.fit()
   
   coeff_seas = res.seasonal[0:12]
   trend = res.trend
   resid = res.resid

   
   m = 1
   nforecast = 6
   tstart = len(y) - nforecast # primo istante da prevederey
   idCoeff1 = tstart % 12
   coeff = np.roll(coeff_seas,-idCoeff1) # Rotate LEFT by idCoeff1 positions (negative means left)
   y = dfpoints.values[:tstart]
   yBase = go_theta(y, m, nforecast)
   yBase = yBase + coeff[:len(yBase)]
   
   diff = dfpoints.values[-6:] - yBase
   rmseBase = np.sqrt(np.dot(diff, diff) / len(yBase))

   y = dfpoints.values[108:-6]
   yFore = go_theta(y, m, nforecast)
   yFore = yFore + coeff[:len(yFore)]
   diff = dfpoints.values[-6:] - yFore
   rmseFore = np.sqrt(np.dot(diff, diff) / len(yBase))
   print(f"full {rmseBase:.2f} fore {rmseFore:.2f}")
   
   plt.figure(figsize=(12,8))
   plt.plot(dfpoints.values)
   plt.plot(range(138,144), yBase,label="Base")
   plt.plot(range(138,144), yFore,label="Fore")
   plt.legend()
   plt.show()

   print("fine")