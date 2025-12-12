import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def plotSol(dfsol,dfdata):
   fig = plt.figure(figsize=(10,8))
   plt.plot(dfdata.iloc[:,1],marker='o',linewidth=0,color='red')
   for _, row in dfsol.iterrows():
      t1, t2, m, q = row['t1'], row['t2'], row['m'], row['q']
      x = [t1, t2]
      y = [m * t1 + q, m * t2 + q]
      plt.plot(x, y, marker='o', label=f'm={m}, q={q}')
   plt.legend()
   plt.title("OLS fit")
   plt.show()
   return

if __name__ == '__main__':
   solfile = "test_sol.csv"
   datafile = "home//test.csv"
   dfsol = pd.read_csv("..//F&Bsegmentation//"+solfile)
   dfdata = pd.read_csv("..//..//data//"+datafile)
   plotSol(dfsol,dfdata)
   print("fine")
