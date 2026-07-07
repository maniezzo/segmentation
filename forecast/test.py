import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.forecasting.theta import ThetaModel
import util
from etstheta import *

if __name__ == "__main__":
   name= "N1906" # "N1930" "N1679" "N1402"
   nameCheck,dfpoints = util.read_series(name)
   df = pd.read_csv("data/"+name+"models.csv",usecols=['0','1','2','3','4'])
   y = dfpoints.values
   n = len(y)
   
   stl = STL(y,
             period=12,
             seasonal=len(y) * 2 - 1,  # Large window forces it to look at all points
             seasonal_deg=0,  # Forces the seasonal component to be strictly periodic
             robust=True)
   res = stl.fit()
   
   coeff_seas = res.seasonal[0:12]
   trend = res.trend
   resid = res.resid

   y = trend+resid
   m = 1
   nforecast = 6
   tstart = len(y) - nforecast # primo istante da prevederey
   idCoeff1 = tstart % 12
   coeff = np.roll(coeff_seas,-idCoeff1) # Rotate LEFT by idCoeff1 positions (negative means left)

   # ---------------------- theta
   yTest = y[:-6]
   yBase = go_theta(yTest, m, nforecast)
   yBase = yBase + coeff[:len(yBase)]
   
   diff = dfpoints.values[-6:] - yBase
   rmseBase = np.sqrt(np.dot(diff, diff) / len(yBase))

   yTest = y[85:-6]
   yFore = go_theta(yTest, m, nforecast)
   yFore = yFore + coeff[:len(yFore)]
   diff = dfpoints.values[-6:] - yFore
   rmseFore = np.sqrt(np.dot(diff, diff) / len(yBase))
   print(f"full {rmseBase:.2f} fore {rmseFore:.2f}")
   
   plt.figure(figsize=(12,8))
   plt.plot(dfpoints.values[-6:])
   plt.plot(yBase,label="Base")
   plt.plot(yFore,label="Fore")
   plt.legend()
   plt.show()

   plt.figure(figsize=(12, 8))
   plt.plot(y[-6:])
   plt.plot(yBase - coeff[:len(yFore)], label="Base")
   plt.plot(yFore - coeff[:len(yFore)], label="Fore")
   plt.legend()
   plt.show()

   # ---------------------- HW
   yTest = y[:-6]
   yBase = go_HW(yTest, m, nforecast)
   yBase = yBase + coeff[:len(yBase)]

   diff = dfpoints.values[-6:] - yBase
   rmseBase = np.sqrt(np.dot(diff, diff) / len(yBase))

   yTest = y[85:-6]
   yFore = go_HW(yTest, m, nforecast)
   yFore = yFore + coeff[:len(yFore)]
   diff = dfpoints.values[-6:] - yFore
   rmseFore = np.sqrt(np.dot(diff, diff) / len(yBase))
   print(f"full {rmseBase:.2f} fore {rmseFore:.2f}")

   plt.figure(figsize=(12, 8))
   plt.plot(dfpoints.values[-6:])
   plt.plot(yBase, label="Base")
   plt.plot(yFore, label="Fore")
   plt.legend()
   plt.show()

   plt.figure(figsize=(12, 8))
   plt.plot(y[-6:])
   plt.plot(yBase - coeff[:len(yFore)], label="Base")
   plt.plot(yFore - coeff[:len(yFore)], label="Fore")
   plt.legend()
   plt.show()

   print("fine")