import numpy as np, pandas as pd, os
import matplotlib.pyplot as plt

def plotSol(dfsol,dfdata):
   ymin = dfdata.iloc[:,1].min()
   ymax = dfdata.iloc[:,1].max()
   yrange = ymax-ymin
   fig, ax = plt.subplots(figsize=(10,6))
   ax.plot(dfdata.iloc[:,1],marker='.',linewidth=0,color='blue')
   for _, row in dfsol.iterrows():
      t1, t2, m, q = row['t1'], row['t2'], row['m'], row['q']
      x = [t1, t2]
      y = [m * t1 + q, m * t2 + q]
      print(t1,t2,x)
      ax.plot(x, y, linewidth=3, color="red", label=f'm={m}')
      ax.vlines(x=t2, ymin=0, ymax=ymax+0.5*yrange, ls='dashed', color="lightgrey")
   #plt.legend()
   plt.ylim(ymin-0.5*yrange, ymax+0.5*yrange)
   plt.title(name)
   plt.savefig(f"{name}.eps", format="eps")
   plt.show()
   return

if __name__ == '__main__':
   os.chdir(os.path.dirname(os.path.abspath(__file__)))
   solfile = "test_sol.csv"
   dfsol = pd.read_csv("..//F&Bsegmentation//"+solfile)
   datafile = dfsol.columns[5].replace("/","//")
   dfdata = pd.read_csv("..//..//data//"+datafile+".csv")
   name = datafile.split("//", 1)[-1]
   name = name.removesuffix(".csv")
   name = name + " " + dfsol.columns[4]
   plotSol(dfsol,dfdata)
   print("fine")
