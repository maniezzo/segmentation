import pandas as pd, util, time
import numpy as np
from optGurobi import go_Gurobi
from Dinkelbach import solve_spp_average_cost
from etstheta import go_HW,go_theta
from statsmodels.tsa.seasonal import STL

def go_opt(name, dfModels, dfpoints, isTheta):
   tstart = time.time()
   y = dfpoints.values
   # -------------------------- soluzione AIC, no forecast
   isAIC = False
   if(isAIC):
      maxNseg = 10 # max numero di segmenti in soluzione
      lstVar = go_Gurobi(dfModels, maxNseg)

   # ----------------------- soluzione rmse forecast
   isForecast = True
   if(isForecast):
      lstIntervals = []
      lstCosts = []
      for i in range(len(dfModels)):
         lstIntervals.append((dfModels.iloc[i, 0], dfModels.iloc[i, 1]))
         lstCosts.append(dfModels.iloc[i, 3])
      nrows = dfModels.iloc[-1, 1] + 1  # partono da 0
      result = solve_spp_average_cost(lstIntervals, lstCosts, nrows=nrows, verbose=True)
      print()
      if result["status"] == "optimal":
         sel = result["selected_sets"]
         print(f"Status        : {result['status']}")
         print(f"Iterations    : {result['iterations']}")
         print(f"Selected sets : {sel}")
         print(f"Costs         : {[lstCosts[j] for j in sel]}")
         print(f"Average cost  : {result['lambda_star']:.6f}")
         print(f"Check         : {sum(lstCosts[j] for j in sel) / len(sel):.6f}")
      else:
         print(f"Status: {result['status']}")
      lstVar = sel
   # ----------------------- results output section
   tend = time.time()
   tcpu = tend-tstart
   
   if(isTheta):
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
      tstart = len(y) - nforecast  # primo istante da prevederey
      idCoeff1 = tstart % 12
      coeff = np.roll(coeff_seas, -idCoeff1)  # Rotate LEFT by idCoeff1 positions (negative means left)
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

   util.plotSol(name, lstVar, dfModels, dfpoints, yBase, yFore, "theta")
   print(f'fine, t.cpu = {tcpu:.2f}')
   print(f"RMSE base = {rmseBase:.2f}, RMSE fore = {rmseFore:.2f}")

if __name__ == "__main__":
   name = "N1930"  # "N1679" "N1402"
   nameCheck, dfpoints = util.read_series(name)
   dfModels = pd.read_csv("data/" + name + "models.csv", usecols=['0', '1', '2', '3', '4'])
   isTheta = True  # true: theta, false:HW

   go_opt(dfModels, dfpoints, isTheta)
   print("fine")
