import pandas as pd, util, time
from optGurobi import go_Gurobi
from Dinkelbach import solve_spp_average_cost
from etstheta import go_HW,go_theta
import numpy as np

if __name__ == "__main__":
   name="N1930" #"N1679" "N1402"
   nameCheck,dfpoints = util.read_series(name)
   df = pd.read_csv("data/"+name+"models.csv",usecols=['0','1','2','3','4'])
   isTheta = True # true: theta, false:HW
   tstart = time.time()

   # -------------------------- soluzione AIC, no forecast
   isAIC = False
   if(isAIC):
      maxNseg = 10 # max numero di segmenti in soluzione
      lstVar = go_Gurobi(df,maxNseg)

   # ----------------------- soluzione rmse forecast
   isForecast = True
   if(isForecast):
      lstIntervals = []
      lstCosts = []
      for i in range(len(df)):
         lstIntervals.append((df.iloc[i, 0], df.iloc[i, 1]))
         lstCosts.append(df.iloc[i, 3])
      nrows = df.iloc[-1, 1] + 1  # partono da 0
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
      m = 12
      y = dfpoints.values[:-6]
      nforecast = 6
      yBase = go_theta(y,m,nforecast)
      diff = dfpoints.values[-6:] - yBase
      rmseBase = np.sqrt(np.dot(diff, diff) / len(yBase))
      
      lastChange = df.iloc[lstVar[-1],0]
      y = y[lastChange:]
      yfore = go_theta(y,1,nforecast)
      diff = dfpoints.values[-6:] - yfore
      rmseFore = np.sqrt(np.dot(diff, diff) / len(yfore))

   util.plotSol(name, lstVar,df,dfpoints, yBase, yfore, "theta")
   print(f'fine, t.cpu = {tcpu:.2f}')
   print(f"RMSE base = {rmseBase:.2f}, RMSE fore = {rmseFore:.2f}")
   print("fine")
