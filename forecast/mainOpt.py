import pandas as pd, numpy as np, time
from optGurobi import go_Gurobi,plotSol

def read_series(name):
   df = pd.read_csv("C:\git\segmentation\data\M3\M3month.csv")
   idSeries = np.where(df['Series'] == name)[0][0]
   ds = pd.to_numeric(df.iloc[idSeries,6:].dropna(), errors="coerce")
   return df.iloc[idSeries,0],ds

if __name__ == "__main__":
   name="N1930" #"N1679" "N1402"
   nameCheck,dfpoints = read_series(name)
   df = pd.read_csv("data/"+name+"models.csv",usecols=['0','1','2','3','4'])
   maxNseg = 10 # max numero di segmenti in soluzione
   tstart  = time.time()
   lstVar = go_Gurobi(df,maxNseg)
   tend = time.time()
   tcpu = tend-tstart
   # ----------------------- results output section
   plotSol(name, lstVar,df,dfpoints,"theta")
   print(f'fine, t.cpu = {tcpu:.2f}')
   print("fine")
