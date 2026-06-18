import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import ast

# SPP model, low segments endpoint, high upper endpoint, cost segment cost
def go_Gurobi(df, maxNseg):
   low  = df.iloc[:,0].values
   high = df.iloc[:,1].values
   cost = df.iloc[:,2].values
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

# plot della soluzione: nome serie, lista var in sol, df segmenti, df punti serie
def plotSol(name, lstVar, dfdata, dfpoints):
   ymin = dfpoints.min()
   ymax = dfpoints.max()
   yrange = ymax - ymin
   y = dfpoints.values
   m=0
   
   fig, ax = plt.subplots(figsize=(10, 6))
   ax.plot(dfpoints, marker='.', linewidth=0, color='blue')
   for i in lstVar:
      t1, t2, model = dfdata.iloc[i, 0], dfdata.iloc[i, 1], dfdata.iloc[i, 4]
      model = ast.literal_eval(model)
      x = range(t1, t2)
      ypred = reconstruct_arima(t1,t2,model,y)
      print(t1, t2, x)
      ax.plot(x, ypred, linewidth=3, color="red", label=f'm={m}')
      ax.vlines(x=t2, ymin=0, ymax=ymax + 0.5 * yrange, ls='dashed', color="lightgrey")
   # plt.legend()
   plt.ylim(ymin - 0.5 * yrange, ymax + 0.5 * yrange)
   plt.title(name)
   plt.savefig(f"data/{name}.eps", format="eps")
   plt.show()
   return
