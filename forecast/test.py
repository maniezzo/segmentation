import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import util
from models import *
from Dinkelbach import solve_spp_average_cost

if __name__ == "__main__":
   with open("test_instances.txt", "r") as f:
      for line in f:
         name = line.strip()
         
         # Stop at first empty row
         if not name:
            break
         else:
            filename = name
         
   print(filename)
   nameCheck, dfpoints, m, validation, min, max = util.read_series(filename)
   dfModels = pd.read_csv("data/" + nameCheck + f"models_{validation}.csv",
                          usecols=["t1", "t2", "AR1", "HW", "theta", "RF"], )
   yorg = np.asarray(dfpoints, dtype=float)
   nforecast = validation # letto in input

   coeff_seas,trend,resid = util.deseason(yorg,m=m)
   y = trend+resid                            # dati destagionalizzati
   n = len(y)
   
   tstart = len(y) - nforecast  # primo istante da prevedere
   #idCoeff1 = tstart % m
   #coeff = np.roll(coeff_seas, -idCoeff1)  # Rotate LEFT by idCoeff1 positions (negative means left)
   #reps = (nforecast + len(coeff) - 1) // len(coeff)  # Calculate needed repetitions, in case nforecast > coeff
   #coeff = np.tile(coeff, reps)  # Vectorized concatenation/tiling, in case nforecast > coeff
   coeff = coeff_seas[tstart:]

   lstIntervals = [] # lista degli intervalli dei segmenti
   for i in range(len(dfModels)): lstIntervals.append((dfModels.iloc[i, 0], dfModels.iloc[i, 1]))
   
   modelNames = ["AR1","HW","theta","RF"]
   idModel = 2    # theta
   
   model = modelNames[idModel]
   lstCosts = []
   for i in range(len(dfModels)):
      lstCosts.append(dfModels.iloc[i, idModel + 2])  # dopo t1 e t2
   nrows = dfModels.iloc[-1, 1] + 1  # righe partono da 0
   result = solve_spp_average_cost(lstIntervals, lstCosts, nrows=nrows, verbose=True)
   print()
   if result["status"] == "optimal":
      sel = result["selected_sets"]
      print(f"Model         : {model}")
      print(f"Status        : {result['status']}")
      print(f"Iterations    : {result['iterations']}")
      print(f"Selected sets : {sel}")
      print(f"Costs         : {[lstCosts[j] for j in sel]}")
      print(f"Average cost  : {result['lambda_star']:.6f}")
      print(f"Check         : {sum(lstCosts[j] for j in sel) / len(sel):.6f}")
   else:
      print(f"Status: {result['status']}")
   lstVar = sel
   
   # ------------------------------------- theta
   if (idModel == 2):
      lstVarTh = lstVar[:]
      yTrain = yorg[:tstart]
      _, yBase = go_theta(yTrain, nforecast, m)
      #yBase = yBase + coeff[:len(yBase)] # aggiungo stagionalità addittiva
      diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yBase
      rmseBaseTh = np.sqrt(np.dot(diff, diff) / len(yBase))
      
      lastInt = lstVar[-1]
      t0 = lstIntervals[lastInt][0]
      yTest = yorg[t0:tstart]
      _, yFore = go_theta(yTest, nforecast, m)
      #yFore = yFore + coeff[:len(yFore)]
      diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yFore
      rmseForeTh = np.sqrt(np.dot(diff, diff) / len(yFore))
      print(f"Theta: full {rmseBaseTh:.2f} fore {rmseForeTh:.2f}")
   
   
   plt.figure(figsize=(12,8))
   plt.plot(dfpoints.values[-nforecast:])
   plt.plot(yBase,label="Base")
   plt.plot(yFore,label="Fore")
   plt.title("forecasts")
   plt.legend()
   plt.show()

   plt.figure(figsize=(12, 8))
   plt.plot(y[-nforecast:])
   plt.plot(yBase - coeff[:len(yFore)], label="Base")
   plt.plot(yFore - coeff[:len(yFore)], label="Fore")
   plt.title("destagionalizzati")
   plt.legend()
   plt.show()

   print("fine")