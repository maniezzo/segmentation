import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import ast
import numpy as np
from scipy.stats import linregress

# SPP model, low segments endpoint, high upper endpoint, cost segment cost
def go_Gurobi(df, maxNseg):
   low  = df.iloc[:,0].values
   high = df.iloc[:,1].values
   cost = df.iloc[:,3].values
   nseg = len(cost)  # num of segments
   npoints = high[-1]  # num of points to cover
   status_dict = {
      GRB.OPTIMAL: "OPTIMAL",
      GRB.INFEASIBLE: "INFEASIBLE",
      GRB.UNBOUNDED: "UNBOUNDED",
      GRB.LOADED: "LOADED"
   }
   sol = []

   m = gp.Model("SPPcover")
   x = m.addVars(nseg, vtype=GRB.CONTINUOUS, lb=0, ub=1, name="x")  # linear relaxation

   # Objective
   m.setObjective(gp.quicksum(cost[i] * x[i] for i in range(nseg)), GRB.MINIMIZE)

   # Constraints, partitioning
   constrs = {}
   for j in range(npoints):
      constrs[j] = m.addConstr(gp.quicksum(x[i] for i in range(nseg) if (j >= low[i] and j <= high[i])) == 1,
                               name=f"cover_{j}")

   m.optimize()
   print("LP Status:", status_dict.get(m.status, f"Status {m.status}"))

   # LP Duals (Pi) and Reduced Costs (RC)
   print("LP Objective:", m.ObjVal)
   '''
   for j in range(npoints):
      print(f"Dual for constraint {j}: {constrs[j].Pi}")
   for i in range(nseg):
      print(f"Reduced cost x[{i}]: {x[i].RC}")
   '''

   # IP, integer version
   for var in x.values():
      var.vtype = GRB.BINARY
   m.optimize()
   print("IP Status:", status_dict.get(m.status, f"Status {m.status}"))

   # Add cardinality constraint (same code if after LP)
   m.addConstr(gp.quicksum(x[i] for i in range(nseg)) <= maxNseg, name=f"maxNseg")
   if (npoints < 20):
      m.write("SPPcover.lp")

   m.optimize()  # re-solve with added constraint
   print("New IP Status:", status_dict.get(m.status, f"Status {m.status}"))

   if m.status == GRB.OPTIMAL:
      print("Solution feasible and optimal")
      print("MIP Objective:", m.ObjVal)
      # Optimal values
      for i in range(nseg):
         if (x[i].X > 0):
            sol.append(i)
   elif m.status == GRB.INFEASIBLE:
      print("Problem is infeasible")
      m.computeIIS()
      m.write("model.ilp")  # .ilp shows conflicting constraints
   elif m.status == GRB.UNBOUNDED:
      print("Problem is unbounded")
   else:  # ???
      print(f"Status: {m.status} - {gp.GRB.status[m.status]}")

   return sol

def reconstruct_arima(t1,t2,model,y):
   y_train = y[t1:t2]
   # Fit ARIMA
   model = ARIMA(y_train, order=model)
   fitted = model.fit()
   
   # predicted values (IN-SAMPLE)
   pred = fitted.predict()
   return pred

# home-made reconstruction, it might be wrong
def reconstruct_theta(t1, t2, y, m):
   y_train = np.array(y[t1:t2], dtype=float)
   n = len(y_train)
   
   theta_model = ThetaModel(y_train, period=m)
   fitted = theta_model.fit()
   
   alpha = fitted.params['alpha']
   b0 = fitted.params['b0']
   
   # Replicate multiplicative deseasonalization
   cma = np.convolve(y_train, np.ones(m) / m, mode='valid')
   offset = m // 2
   ratios = np.array([y_train[i + offset] / cma[i] for i in range(len(cma))])
   
   seas_factors = np.ones(m)
   for s in range(m):
      idx = [i for i in range(len(ratios)) if (i + offset) % m == s % m]
      if idx:
         seas_factors[s] = np.mean([ratios[i] for i in idx])
   seas_factors /= seas_factors.mean()
   
   sf_full = np.array([seas_factors[t % m] for t in range(n)])
   deseas = y_train / sf_full
   
   # Recover b1 via OLS on deseasonalized series (as statsmodels does internally)
   t_idx = np.arange(n)
   b1, _, _, _, _ = linregress(t_idx, deseas)
   # b0 from params is the statsmodels intercept; use it directly
   trend = b0 + b1 * t_idx
   
   # SES on deseasonalized series (one-step-ahead)
   L = np.empty(n)
   L[0] = deseas[0]
   for t in range(1, n):
      L[t] = alpha * deseas[t - 1] + (1 - alpha) * L[t - 1]
   
   # Theta combination and reseasonalize
   combined = 0.5 * (L + trend)
   in_sample_preds = combined * sf_full
   
   return in_sample_preds

# another home-made reconstruction for theta
def reconstruct_theta2(t1, t2, y, m):
   burnin = m
   if(m<6): burnin=6  # proprio il minimo numero di osservazioni
   preds = [None] * burnin
   period = max(1,m)
   
   for i in range(t1 + burnin, t2):
      y_train = y[t1:i]
      fitted = ThetaModel(y_train, period=period).fit()
      pred = fitted.forecast(steps=1).iloc[0]
      preds.append(pred)
   
   return preds

def reconstruct_HW():
   return

# plot della soluzione: nome serie, lista var in sol, df segmenti, df punti serie
def plotSol(name, lstVar, dfdata, dfpoints, model):
   ymin = dfpoints.min()
   ymax = dfpoints.max()
   yrange = ymax - ymin
   y = dfpoints.values.ravel()
   m=0
   
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(dfpoints, marker='.', linewidth=0, color='blue')
   for i in lstVar:
      t1, t2, dfmodel = dfdata.iloc[i, 0], dfdata.iloc[i, 1], dfdata.iloc[i, 4]
      x = range(t1, t2)
      if(model=="theta"):
         ypred = reconstruct_theta2(t1,t2,y,m)
      elif(model=="HW"):
         ypred = reconstruct_HW(t1, t2, y, m)
      elif(model=="ARIMA"):
         model = ast.literal_eval(dfmodel)
         ypred = reconstruct_arima(t1,t2,model,y)
      else:
         print("No acceptable model")
         return
      print(t1, t2, x)
      ax.plot(x, ypred, linewidth=3, color="red", label=f'm={m}')
      ax.vlines(x=t2, ymin=0, ymax=ymax + 0.5 * yrange, ls='dashed', color="lightgrey")
   # plt.legend()
   plt.ylim(ymin - 0.5 * yrange, ymax + 0.5 * yrange)
   plt.title(name)
   plt.savefig(f"data/{name}.eps", format="eps")
   plt.show()
   return
