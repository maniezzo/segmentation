'''
using ruptures for change point detection
'''
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import ruptures as rpt
from math import log
from ruptures.base  import BaseCost
from ruptures.costs import NotEnoughPoints

class LogLikCost(BaseCost):
    """Custom cost for exponential signals."""
    # The 2 following attributes must be specified for compatibility.
    model = ""
    min_size = 2

    def fit(self, signal):
        """Set the internal parameter."""
        self.signal = signal
        return self

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        sub = self.signal[start:end]
        return (end - start) * log(sub.mean()) # negative log-likelihood,


class RMSECost(BaseCost):
   """Custom cost class implementing RMSE as cost function."""
   # The 2 following attributes must be specified for compatibility.
   model = ""
   min_size = 2

   def fit(self, signal):
      """Set the internal parameter."""
      self.signal = signal
      return self

   # Define the error method
   def error(self, start, end):
      if end - start < 2:
         raise NotEnoughPoints
      segment = self.signal[start:end]
      rmse = np.sqrt(np.mean((segment - np.mean(segment)) ** 2))
      return rmse

def printResults(method,results,y):
   f = open(f"res_{method}.csv", "w")
   f.write("id, low, hi, m, q, cost\n"),
   end1=0
   for i in range(len(results)):
      # linear regression
      coeff = np.polyfit(range(end1,results[i]),y[end1:results[i]],1)
      m = coeff[0]
      q = coeff[1]
      f.write(f"{i},{end1},{results[i]},{m},{q}\n")
      end1=results[i]
   f.close()
   return

def go_ruptures(y):
   # model, min_size, pen are hyperparameters
   dim = 1
   n_samples = len(y)
   signal = np.atleast_2d(np.array(y)).T # convert to ruptures-compatible internal format

   # detection, PELT (Pruned Exact Linear Time)
   method = "PELT"
   algo = rpt.Pelt(model="l2", min_size=20).fit(signal)
   # predict with penalty value (here 10) to get the change points
   result = algo.predict(pen=1)
   numBreakPnts = len(result)
   # display
   # signal: time series data, bkps list of previously known change points, result list of change points detected by the algorithm
   bkps = []
   rpt.display(signal, bkps, result)
   plt.title(method)
   plt.show()
   print(f"Pelt: {result}")
   printResults(method,result,y)

   # dynamic programming, needs the number of points
   algo = rpt.Dynp(model="l2", min_size=20).fit(signal)
   result = algo.predict(n_bkps=numBreakPnts)
   rpt.display(signal, bkps, result)
   plt.title("DynProgr")
   plt.show()
   print(f"DynProgr: {result}")

   # rolling window
   algo = rpt.Window(model="l2", width=20)
   algo.fit(signal)
   result = algo.predict(n_bkps=numBreakPnts)
   rpt.display(signal, bkps, result)
   plt.title("Rolling")
   plt.show()
   print(f"Rolling: {result}")

   # bottom up
   algo = rpt.BottomUp(model="l2", min_size=20)
   algo.fit(signal)
   result = algo.predict(n_bkps=numBreakPnts)
   rpt.display(signal, bkps, result)
   plt.title("Bottom up")
   plt.show()
   print(f"Bottom up: {result}")

   # kernel (linear, rbf, cosine)
   fig, ax = plt.subplots(3, 1, figsize=(1280 / 96, 720 / 96), dpi=96)
   for i, kernel in enumerate(['linear', 'rbf', 'cosine']):
      algo = rpt.KernelCPD(kernel=kernel, min_size=20)
      algo.fit(signal)
      try:
         result = algo.predict(n_bkps=numBreakPnts)
         ax[i].plot(signal)
         for bkp in result:
            ax[i].axvline(x=bkp, color='k', linestyle='--')
         ax[i].set_title(f"Kernel model with {kernel} kernel")
         print(f"Kernel {kernel}: {result}")
      except:
         print(f"kernel {kernel} cannot be applied")
   fig.tight_layout()
   plt.show()

   # custom cost functions
   algo = rpt.Pelt(custom_cost=LogLikCost()).fit(signal)
   result = algo.predict(pen=10)
   rpt.display(signal, bkps, result)
   plt.title("Negative log-likelihood")
   plt.show()
   print(f"Negative log-likelihood: {result}")

   algo = rpt.Dynp(custom_cost=RMSECost()).fit(signal)
   result = algo.predict(n_bkps=numBreakPnts)
   rpt.display(signal, bkps, result)
   plt.title("RMSE")
   plt.show()
   print(f"RMSE: {result}")

def main():
   dataset = "PTemp_C_Avg"  #"BTC-USD.csv")  "test.csv");  PTemp_C_Avg
   df = pd.read_csv(f"..//{dataset}.csv");
   y = df.iloc[:,1]#.values[:100]
   x = np.arange(len(y))

   go_ruptures(y)

if __name__ == "__main__":
   main()
   print("fine")
   exit()