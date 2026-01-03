import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def plotSol(dfsol,dfdata):
   fig = plt.figure(figsize=(10,6))
   plt.plot(dfdata.iloc[:,1],marker='.',linewidth=0,color='red')
   for _, row in dfsol.iterrows():
      t1, t2, m, q = row['t1'], row['t2'], row['m'], row['q']
      if(t1>0): t1+=1;  # segmenti senza estremi in comune
      x = [t1, t2]
      y = [m * t1 + q, m * t2 + q]
      plt.plot(x, y, linewidth=2, color="blue", label=f'm={m}')
   #plt.legend()
   plt.title(name)
   plt.savefig(f"{name}.eps", format="eps")
   plt.show()
   return

if __name__ == '__main__':
   solfile = "test_sol.csv"
   datafile = "M3//N2834.csv"
   dfsol = pd.read_csv("..//F&Bsegmentation//"+solfile)
   dfdata = pd.read_csv("..//..//data//"+datafile)
   name = datafile.split("//", 1)[-1]
   name = name.removesuffix(".csv")
   name = name + " " + dfsol.columns[4]
   plotSol(dfsol,dfdata)
   print("fine")
