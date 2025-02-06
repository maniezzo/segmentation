import numpy as np
import pandas as pd,os
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, nbinom
import nelder_mead as nm
import pmdarima as pm
from PSO import PSO

def covid_nbinom(pars):
    return pars[2] * nbinom.pmf(xval, pars[0], pars[1])
    
def loss_nbinom(pars):
    return ((y - covid_nbinom(pars)) ** 2).replace(np.nan, np.inf).sum()

def fit_nbinom(xval):
   PSOopt = PSO(nump=50, objFunc=loss_nbinom, ndim=3, maxiter=5000, numneigh=5,
                 x_min=np.array([10, 0.1, 100000.]), x_max=np.array([1000, 100, 10000000.]),
                 #x_min=np.array([10, 0.1, 109000.]), x_max=np.array([1000, 100, 10000000.]),
                 isMax=False, c0=0.25, c1=2, c2=2,isVerbose=True)
   #x_min = np.array([0.1, 0.1, 50000.]), x_max = np.array([0.5, 0.5, 200000.]),
   res = PSOopt.runPSO()
   nbinom_best = nm.nelder_mead(loss_nbinom, res)
   print(nbinom_best)
   nbinom_rmse = np.sqrt(nbinom_best[1] / len(xval))
   return nbinom_best,nbinom_rmse

if __name__ == "__main__":
   # change working directory to script path
   abspath = os.path.abspath(__file__)
   dname = os.path.dirname(abspath)
   os.chdir(dname)
   idColumn = 3
   isBase = False
   ytot = []

   #initialize plot
   fig = plt.figure()
   plt.title("Negative binomial fit")

   for idColumn in range(3,8):
      if(isBase):
         df = pd.read_csv("../Covid_italia_22.csv")
         data = df.iloc[:120, 1].rolling(7).mean()
         y = data[30:]
      else:
         df = pd.read_csv("../Covid_runs_22.csv")
         data = df.iloc[:, idColumn].dropna()
         y = df.iloc[:, idColumn].dropna()

      n0 = len(ytot)
      ytot = np.append(ytot, y)
      print(f"Added {len(y)} values, tot {len(ytot)}")
      xval = np.arange(len(y))
      miny = min(y)
      y = y - miny

      trendhorizon = 0

      # initialize plot
      #plt.title(df.columns[idColumn])
      #plt.plot(xval,y + miny, label="Original data")

      nbinom_best,nbinom_rmse = fit_nbinom(xval)
      xplot = range(len(xval))+n0*np.ones(len(xval))
      plt.plot(xplot,nbinom_best[0][2] * nbinom.pmf(np.arange(0, len(y) + trendhorizon),
                     nbinom_best[0][0],nbinom_best[0][1]) + miny,label=df.columns[idColumn],
                     linewidth=6)

      print("\n           Root Mean Square Error (RMSE)")
      print('{0:<20}: {1}'.format("Negative binomial", nbinom_rmse))

   plt.plot(ytot,"--",linewidth=5,color="black",label="actual data")
   plt.plot(ytot,linewidth=2,color="black")
   plt.grid(axis='y')
   plt.legend(loc="upper right")
   plt.savefig("negativeBinFit.eps",format="eps")
   plt.show()

   fig = plt.figure()
   plt.title("Negative binomial fit")
   plt.plot(ytot, "--", linewidth=5, color="black", label="actual data")
   plt.plot(ytot, linewidth=2, color="black")
   plt.grid(axis='y')
   plt.legend(loc="upper right")
   plt.savefig(f"negativeBinFit0.eps", format="eps")
   plt.show()

   #progressive plot
   for stage in range(4,9):
      fig = plt.figure()
      plt.title("Negative binomial fit")
      plt.plot(ytot,"--",linewidth=5,color="black",label="actual data")
      plt.plot(ytot,linewidth=2,color="black")
      p0 = 0
      p1 = 0

      for idColumn in range(3,stage):
         if(isBase):
            df = pd.read_csv("../Covid_italia_22.csv")
            data = df.iloc[:120, 1].rolling(7).mean()
            y = data[30:]
         else:
            df = pd.read_csv("../Covid_runs_22.csv")
            data = df.iloc[:, idColumn].dropna()
            y = df.iloc[:, idColumn].dropna()

         n0 = len(ytot)
         print(f"Added {len(y)} values, tot {len(ytot)}")
         p0,p1=p1,p1+len(y)
         xval = np.arange(len(y))
         miny = min(y)
         y = y - miny

         trendhorizon = 0

         nbinom_best,nbinom_rmse = fit_nbinom(xval)
         xplot = range(p0,p1) #range(len(xval))+n0*np.ones(len(xval))
         yplot = nbinom_best[0][2] * nbinom.pmf(np.arange(0, len(y) + trendhorizon),
                        nbinom_best[0][0],nbinom_best[0][1]) + miny
         plt.plot(xplot,yplot,label=df.columns[idColumn],linewidth=6)

      plt.grid(axis='y')
      plt.legend(loc="upper right")
      plt.savefig(f"negativeBinFit{stage}.eps",format="eps")
      plt.show()
      print("Figura")

