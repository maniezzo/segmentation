import pandas as pd, numpy as np, time
from optGurobi import go_Gurobi,plotSol
from Dinkelbach import solve_spp_average_cost

def read_series(name):
   df = pd.read_csv("C:\git\segmentation\data\M3\M3month.csv")
   idSeries = np.where(df['Series'] == name)[0][0]
   ds = pd.to_numeric(df.iloc[idSeries,6:].dropna(), errors="coerce")
   return df.iloc[idSeries,0],ds

if __name__ == "__main__":
   name="N1930" #"N1679" "N1402"
   nameCheck,dfpoints = read_series(name)
   df = pd.read_csv("data/"+name+"models.csv",usecols=['0','1','2','3','4'])
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
   plotSol(name, lstVar,df,dfpoints,"theta")
   print(f'fine, t.cpu = {tcpu:.2f}')
   print("fine")
