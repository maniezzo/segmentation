import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def plotSol(dfsol,dfdata):
   ymax = dfdata.iloc[:,1].max()
   fig, ax = plt.subplots(figsize=(10,6))
   ax.plot(dfdata.iloc[:,1],marker='.',linewidth=0,color='red')
   for _, row in dfsol.iterrows():
      t1, t2, m, q = row['t1'], row['t2'], row['m'], row['q']
      x = [t1, t2]
      y = [m * t1 + q, m * t2 + q]
      print(t1,t2,x)
      ax.plot(x, y, linewidth=2, color="blue", label=f'm={m}')
      ax.vlines(x=t2, ymin=0, ymax=ymax, ls='dashed', color="lightgrey")
   #plt.legend()
   plt.title(name)
   plt.savefig(f"{name}.eps", format="eps")
   plt.show()
   return

if __name__ == '__main__':
   solfile = "test_sol.csv"
   dfsol = pd.read_csv("..//F&Bsegmentation//"+solfile)
   datafile = dfsol.columns[5].replace("/","//")
   dfdata = pd.read_csv("..//..//data//"+datafile+".csv")
   name = datafile.split("//", 1)[-1]
   name = name.removesuffix(".csv")
   name = name + " " + dfsol.columns[4]
   plotSol(dfsol,dfdata)
   print("fine")
