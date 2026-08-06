import pandas as pd, util, time
import numpy as np
from optGurobi import go_Gurobi
from Dinkelbach import solve_spp_average_cost
from models import go_HW,go_theta,go_RF,go_AR1

def go_opt(name, dfModels, dfpoints, m, nforecast):
   # m; stagionalità serie originale (12 o 4)
   isAIC      = False
   isForecast = True    # segmentazione basata su efficacia predittiva

   tstart = time.time()
   y = np.asarray(dfpoints, dtype=float)
   coeff_seas,trend,resid = util.deseason(y,m=m)
   y = trend+resid                            # dati destagionalizzati
   
   tstart   = len(y) - nforecast  # primo istante da prevedere
   idCoeff1 = tstart % m
   coeff    = np.roll(coeff_seas, -idCoeff1)  # Rotate LEFT by idCoeff1 positions (negative means left)
   
   # -------------------------- soluzione AIC, no forecast. Backward compatibility (if it worked)
   if(isAIC):
      maxNseg = 10 # max numero di segmenti in soluzione
      lstVar = go_Gurobi(dfModels, maxNseg)

   # ----------------------- soluzione SPP rmse forecast, costo medio
   lstIntervals = []
   for i in range(len(dfModels)): lstIntervals.append((dfModels.iloc[i, 0], dfModels.iloc[i, 1]))
   
   modelNames = ["AR1","HW","theta","RF"]
   for idModel in range(len(modelNames)):
      model = modelNames[idModel]
      lstCosts = []
      for i in range(len(dfModels)):
         lstCosts.append(dfModels.iloc[i, idModel+2]) # dopo t1 e t2
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

      # ----------------------- results output section
      tend = time.time()
      tcpuSeg = tend-tstart
      if (idModel == 0):  # ------------------------------ AR1
         lstVarAR1 = lstVar[:]
         yTrain = y[:tstart]
         hwfit, yBase = go_AR1(yTrain, 1, nforecast)
         yBase = yBase + coeff[:len(yBase)]
         diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yBase
         rmseBaseAR1 = np.sqrt(np.dot(diff, diff) / len(yBase))
         
         lastInt = lstVar[-1]
         t0 = lstIntervals[lastInt][0]
         yTest = y[t0:-nforecast]
         hwfit, yFore = go_AR1(yTest, 1, nforecast)
         yFore = yFore + coeff[:len(yFore)]
         diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yFore
         rmseForeAR1 = np.sqrt(np.dot(diff, diff) / len(yFore))
         print(f"HW: full {rmseBaseAR1:.2f} fore {rmseForeAR1:.2f}")
         util.plotSol(name, lstVarAR1, dfModels, dfpoints, yBase, yFore, m, model=model)
      
      if (idModel == 1):  # ------------------------------ HW
         lstVarHW = lstVar[:]
         yTrain = y[:tstart]
         hwfit, yBase = go_HW(yTrain, 1, nforecast)
         yBase = yBase + coeff[:len(yBase)]
         diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yBase
         rmseBaseSLS = np.sqrt(np.dot(diff, diff) / len(yBase))
         
         lastInt = lstVar[-1]
         t0 = lstIntervals[lastInt][0]
         yTest = y[t0:-nforecast]
         hwfit, yFore = go_HW(yTest, 1, nforecast)
         yFore = yFore + coeff[:len(yFore)]
         diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yFore
         rmseForeSLS = np.sqrt(np.dot(diff, diff) / len(yFore))
         print(f"HW: full {rmseBaseSLS:.2f} fore {rmseForeSLS:.2f}")
         util.plotSol(name, lstVarHW, dfModels, dfpoints, yBase, yFore, m, model=model)

      # ------------------------------------- theta
      if(idModel==2):
         lstVarTh = lstVar[:]
         yTrain = y[:tstart]
         _,yBase = go_theta(yTrain, 1, nforecast)
         yBase = yBase + coeff[:len(yBase)]
         diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yBase
         rmseBaseTh = np.sqrt(np.dot(diff, diff) / len(yBase))
   
         lastInt = lstVar[-1]
         t0 = lstIntervals[lastInt][0]
         yTest = y[t0:-nforecast]
         _,yFore = go_theta(yTest, 1, nforecast)
         yFore = yFore + coeff[:len(yFore)]
         diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yFore
         rmseForeTh = np.sqrt(np.dot(diff, diff) / len(yFore))
         print(f"Theta: full {rmseBaseTh:.2f} fore {rmseForeTh:.2f}")
         util.plotSol(name, lstVarTh, dfModels, dfpoints, yBase, yFore, m, model=model)
         
      if (idModel == 3):  # ------------------------------ RF
         lstVarRF = lstVar[:]
         yTrain = y[:tstart]
         hwfit, yBase = go_RF(yTrain, 1, nforecast)
         yBase = yBase + coeff[:len(yBase)]
         diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yBase
         rmseBaseRF = np.sqrt(np.dot(diff, diff) / len(yBase))
         
         lastInt = lstVar[-1]
         t0 = lstIntervals[lastInt][0]
         yTest = y[t0:-nforecast]
         hwfit, yFore = go_RF(yTest, 1, nforecast)
         yFore = yFore + coeff[:len(yFore)]
         diff = np.asarray(dfpoints, dtype=float)[-nforecast:] - yFore
         rmseForeRF = np.sqrt(np.dot(diff, diff) / len(yFore))
         print(f"RF: full {rmseBaseRF:.2f} fore {rmseForeRF:.2f}")
         util.plotSol(name, lstVarRF, dfModels, dfpoints, yBase, yFore, m, model=model)
   
   print(f'fine, t.cpu SPP = {tcpuSeg:.2f}')
   print(f"{name}, n.intervals {len(lstVarAR1)} RMSE base hw = {rmseBaseAR1:.2f}, RMSE fore hw = {rmseForeAR1:.2f}")
   print(f"{name}, n.intervals {len(lstVarHW)} RMSE base hw = {rmseBaseSLS:.2f}, RMSE fore hw = {rmseForeSLS:.2f}")
   print(f"{name}, n.intervals {len(lstVarTh)} RMSE base th = {rmseBaseTh:.2f}, RMSE fore th = {rmseForeTh:.2f}")
   print(f"{name}, n.intervals {len(lstVarRF)} RMSE base rf = {rmseBaseRF:.2f}, RMSE fore rf = {rmseForeRF:.2f}")
   with open('data/results.txt', 'a') as f:
      f.write(f"{name},")
      f.write(f" nint {len(lstVarAR1)} RMSE base ar = {rmseBaseAR1:.2f}, RMSE fore ar = {rmseForeAR1:.2f}")
      f.write(f" nint {len(lstVarHW)} RMSE base hw = {rmseBaseSLS:.2f}, RMSE fore hw = {rmseForeSLS:.2f}")
      f.write(f" nint {len(lstVarTh)} RMSE base th = {rmseBaseTh:.2f}, RMSE fore th = {rmseForeTh:.2f} ")
      f.write(f" nint {len(lstVarRF)} RMSE base RF = {rmseBaseRF:.2f}, RMSE fore RF = {rmseForeRF:.2f}\n")

if __name__ == "__main__":
   name = "N1679" # "N1930" "N1679" "N1402"
   nameCheck, dfpoints = util.read_series(name)
   dfModels = pd.read_csv("data/" + name + "models.csv", usecols=['0', '1', '2', '3', '4'])
   isTheta = True  # true: theta, false:HW
   go_opt(dfModels, dfpoints, isTheta)
   print("fine")
