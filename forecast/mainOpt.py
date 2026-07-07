import pandas as pd, util, time
import numpy as np
from optGurobi import go_Gurobi
from Dinkelbach import solve_spp_average_cost
from etstheta import go_HW,go_theta

def go_opt(name, dfModels, dfpoints):
   tstart = time.time()
   y = dfpoints.values
   coeff_seas,trend,resid = util.deseason(y)
   y = trend+resid # dati destagionalizzati
   # -------------------------- soluzione AIC, no forecast
   isAIC = False
   if(isAIC):
      maxNseg = 10 # max numero di segmenti in soluzione
      lstVar = go_Gurobi(dfModels, maxNseg)

   # ----------------------- soluzione SPP rmse forecast, costo medio
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
   tcpuSeg = tend-tstart

   m = 1 # dati destagionalizzati
   nforecast = 6
   tstart = len(y) - nforecast  # primo istante da prevedere
   idCoeff1 = tstart % 12
   coeff = np.roll(coeff_seas, -idCoeff1)  # Rotate LEFT by idCoeff1 positions (negative means left)
   # previsione con theta
   isTheta = True
   if(isTheta):
      yTrain = y[:tstart]
      yBase = go_theta(yTrain, m, nforecast)
      yBase = yBase + coeff[:len(yBase)]
      diff = y[-6:] - yBase
      rmseBaseTh = np.sqrt(np.dot(diff, diff) / len(yBase))

      lastInt = lstVar[-1]
      t0 = lstIntervals[lastInt][0]
      yTest = y[t0:-6]
      yFore = go_theta(yTest, m, nforecast)
      yFore = yFore + coeff[:len(yFore)]
      diff = dfpoints.values[-6:] - yFore
      rmseForeTh = np.sqrt(np.dot(diff, diff) / len(yBase))
      print(f"Theta: full {rmseBaseTh:.2f} fore {rmseForeTh:.2f}")

      util.plotSol(name, lstVar, dfModels, dfpoints, yBase, yFore, "theta")

   isTheta = False  # HW
   if(not isTheta):
      yTrain = y[:tstart]
      yBase = go_HW(yTrain, m, nforecast)
      yBase = yBase + coeff[:len(yBase)]
      diff = y[-6:] - yBase
      rmseBaseSLS = np.sqrt(np.dot(diff, diff) / len(yBase))

      lastInt = lstVar[-1]
      t0 = lstIntervals[lastInt][0]
      yTest = y[t0:-6]
      yFore = go_HW(yTest, m, nforecast)
      yFore = yFore + coeff[:len(yFore)]
      diff = dfpoints.values[-6:] - yFore
      rmseForeSLS = np.sqrt(np.dot(diff, diff) / len(yBase))
      print(f"HW: full {rmseBaseSLS:.2f} fore {rmseForeSLS:.2f}")

      util.plotSol(name, lstVar, dfModels, dfpoints, yBase, yFore, "SLS")


   print(f'fine, t.cpu SPP = {tcpuSeg:.2f}')
   print(f"{name}, n.intervals {len(lstVar)} RMSE base = {rmseBase:.2f}, RMSE fore = {rmseFore:.2f} ")
   with open('data/results.txt', 'a') as f:
      f.write(f"{name}, n.intervals {len(lstVar)} RMSE base = {rmseBase:.2f}, RMSE fore = {rmseFore:.2f}\n")

if __name__ == "__main__":
   name = "N1679" # "N1930" "N1679" "N1402"
   nameCheck, dfpoints = util.read_series(name)
   dfModels = pd.read_csv("data/" + name + "models.csv", usecols=['0', '1', '2', '3', '4'])
   isTheta = True  # true: theta, false:HW
   go_opt(dfModels, dfpoints, isTheta)
   print("fine")
