import pandas as pd, numpy as np
import matplotlib,matplotlib.pyplot as plt
import ruptures as rpt
from ruptures.base  import BaseCost
from ruptures.costs import NotEnoughPoints
from sklearn.linear_model import LinearRegression
import time
from AICcost import AICcost
from QRMSEcost import QRMSELinearCost

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

# minimal code for ruptures
def go_rupturesMinimal(data):
    model = "l2"  # Use L2 norm for cost function (default)
    algo = rpt.Pelt(model=model).fit(data)
    changepoints = algo.predict(pen=50)  # Penalty parameter for segmentation

    # Print the detected changepoints
    print("Detected changepoints:", changepoints)

    rpt.display(data, changepoints)
    plt.show()
    return

# PELT with AIC cost function and piecewise linear model
def go_PELT_AIC(y, isAIC=False):
   # 1. Data Cleaning & Preparation
   y_float = np.array(y).astype(float).flatten()
   n = len(y_float)
   x = np.arange(n).reshape(-1, 1)
   
   # Ruptures needs [features, target]
   # We add a column of ones so the cost function accounts for the intercept
   data = np.column_stack((x, np.ones_like(x), y_float))
   
   pen_value = 7
   if(isAIC):
      # PELT with AIC cost
      cost = AICcost(k_params=2).fit(data)
      algo = rpt.Pelt(custom_cost=cost, min_size=5, jump=1).fit(data)
      result = algo.predict(pen=pen_value)  # AIC penalty is inside the cost
   else:
      # PELT with QRMSE cost
      cost = QRMSELinearCost().fit(data)
      algo = rpt.Pelt(custom_cost=cost, min_size=cost.min_size).fit(data)
      result = algo.predict(pen=pen_value)
   
   # changepoints, indices of the end of each segment
   print(f"Cchangepoints: {result}")
   print(f"Number of segments: {len(result)}")
   return result

def go_ruptures(y):
   # model, min_size, pen are hyperparameters
   dim = 1
   n_samples = len(y)
   #signal = np.atleast_2d(np.array(y)).T # convert to ruptures-compatible internal format
   signal = y

   # detection, PELT (Pruned Exact Linear Time)
   method = "PELT"
   algo = rpt.Pelt(model="l2", min_size=20).fit(signal)
   # predict with penalty value to get the change points
   result = algo.predict(pen=1)
   numBreakPnts = len(result)
   # display
   # signal: time series data, bkps list of previously known change points, result list of change points detected by the algorithm
   bkps = []
   rpt.display(signal, bkps, result)
   plt.title(method)
   plt.show()
   print(f"Pelt: {result}")
   printResults(method,result,np.array(y,dtype=np.float64))

   # dynamic programming, needs the number of points
   algo = rpt.Dynp(model="linear", min_size=5).fit(signal)
   result = algo.predict(n_bkps=numBreakPnts)
   rpt.display(signal, bkps, result)
   plt.title("DynProgr")
   plt.show()
   print(f"DynProgr: {result}")

   # rolling window
   algo = rpt.Window(model="linear", width=20)
   algo.fit(signal)
   result = algo.predict(n_bkps=numBreakPnts)
   rpt.display(signal, bkps, result)
   plt.title("Rolling")
   plt.show()
   print(f"Rolling: {result}")

   # bottom up
   algo = rpt.BottomUp(model="linear", min_size=20)
   algo.fit(signal)
   result = algo.predict(n_bkps=numBreakPnts)
   rpt.display(signal, bkps, result)
   plt.title("Bottom up")
   plt.show()
   print(f"Bottom up: {result}")

   # custom cost functions
   #costf = LogLikCost()
   costf = rpt.costs.CostL2()
   algo = rpt.Pelt(custom_cost=costf).fit(signal)
   result = algo.predict(pen=10)
   rpt.display(signal, bkps, result)
   plt.title("Negative log-likelihood")
   plt.show()
   print(f"Negative log-likelihood: {result}")

   algo = rpt.Dynp(custom_cost=costf).fit(signal)
   result = algo.predict(n_bkps=numBreakPnts)
   rpt.display(signal, bkps, result)
   plt.title("RMSE")
   plt.show()
   print(f"RMSE: {result}")

def writeCsv(y, changepoints, filename="results.csv", isAIC=False):
   y_float = np.array(y).astype(float).flatten()
   n = len(y_float)
   x = np.arange(n).reshape(-1, 1)
   
   # Plotting
   plt.figure(figsize=(12, 6))
   plt.scatter(x, y_float, s=10, color='lightgray', label="Data")
   
   # Iterate through segments to plot the fitted lines
   start_idx = 0
   for cp in changepoints:
      # Define the segment range
      seg_x = x[start_idx:cp+1]
      seg_y = y_float[start_idx:cp+1]
      
      # Fit a local linear regression for visualization
      lr = LinearRegression().fit(seg_x, seg_y)
      y_pred = lr.predict(seg_x)
      
      # Plot the segment line
      plt.plot(seg_x, y_pred, color='red', linewidth=3,
               label="Fitted Segment" if start_idx == 0 else "")
      
      # Vertical line for changepoint
      if cp < n:
         plt.axvline(x=cp, color='blue', linestyle='--', alpha=0.6)
      
      start_idx = cp+1
   
   costName = "QRMSE"
   if(isAIC): costName = "AIC"
   plt.title(f"Piecewise Linear Segments (PELT + {costName})")
   plt.xlabel("Index")
   plt.ylabel("Value")
   plt.legend()
   plt.show()
   
   sigma = np.median(np.abs(np.diff(y_float))) / 0.6745
   sig2 = sigma ** 2
   
   cost = 0
   endpoints = [0] + sorted(list(set(changepoints)))
   if endpoints[-1] != n:
      endpoints.append(n)
   
   segment_data = []
   k = 2  # number of parameters (slope, intercept)
   
   # define each segment
   for i in range(len(endpoints) - 1):
      t1 = endpoints[i]
      if(t1>0): t1+=1 # disjoint endpoints
      t2 = endpoints[i + 1]
      
      seg_x = x[t1:t2+1]
      seg_y = y_float[t1:t2+1]
      
      # Linear Regression: y = mx + q
      model = LinearRegression().fit(seg_x, seg_y)
      m = model.coef_[0]
      q = model.intercept_
      
      # Segment AIC Cost
      y_pred = model.predict(seg_x)
      rss = np.sum((seg_y - y_pred) ** 2)
      # AIC = (RSS / sigma^2) + 2k
      cost_aic = (rss / sig2) + (2 * k)
      cost += cost_aic
      
      segment_data.append({'t1': t1,'t2': t2,'m': m,'q': q,'cost': cost_aic})
      print(f"t1: {t1},t2: {t2},m: {m},q: {q},cost: {cost_aic}")

   # Export data
   df = pd.DataFrame(segment_data)
   df.to_csv(filename, index=False)
   
   print(f"Total cost {cost}. Segment data saved to {filename}")
   return df

if __name__ == "__main__":
    matplotlib.use("TkAgg")
    data = pd.read_csv("../data/M3/N1918.csv",usecols=[1])
    
    start_cpu = time.process_time()
    isAIC = False
    results = go_PELT_AIC(data.values,isAIC=isAIC)
    end_cpu = time.process_time()
    print(f"Total CPU time: {end_cpu - start_cpu:.4f} seconds")

    writeCsv(data,results,"results.csv",isAIC=isAIC)
    #go_ruptures(data.values)
    print("fine")
