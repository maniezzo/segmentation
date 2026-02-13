import numpy as np, pandas as pd, time
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

# SPP model, low segments low andpoint, high upper endpoint, cost segment cost
def run_SPP(name,low,high,cost,naxNseg):
    nseg    = len(cost)  # num of segments
    npoints = high[-1]   # num of points to cover
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
        constrs[j] = m.addConstr(gp.quicksum(x[i] for i in range(nseg) if (j>=low[i] and j<=high[i])) == 1, name=f"cover_{j}")

    m.optimize()
    print("LP Status:", status_dict.get(m.status, f"Status {m.status}"))
    
    # LP Duals (Pi) and Reduced Costs (RC)
    print("LP Objective:", m.ObjVal)
    for j in range(npoints):
        print(f"Dual for constraint {j}: {constrs[j].Pi}")
    for i in range(nseg):
        print(f"Reduced cost x[{i}]: {x[i].RC}")
    
    # IP, integer version
    for var in x.values():
        var.vtype = GRB.BINARY
    m.optimize()
    print("IP Status:", status_dict.get(m.status, f"Status {m.status}"))
    
    # Add cardinality constraint (same code if after LP)
    m.addConstr(gp.quicksum(x[i] for i in range(nseg)) <= naxNseg, name=f"naxNseg")
    if (npoints < 20):
       m.write("SPPcover.lp")

    m.optimize()  # re-solve with added constraint
    print("New IP Status:", status_dict.get(m.status, f"Status {m.status}"))
    
    if m.status == GRB.OPTIMAL:
        print("Solution feasible and optimal")
        print("MIP Objective:", m.ObjVal)
        # Optimal values
        for i in range(nseg):
            if(x[i].X > 0):
                sol.append(i)
    elif m.status == GRB.INFEASIBLE:
        print("Problem is infeasible")
        m.computeIIS()
        m.write("model.ilp")  # .ilp shows conflicting constraints
    elif m.status == GRB.UNBOUNDED:
        print("Problem is unbounded")
    else: # ???
        print(f"Status: {m.status} - {gp.GRB.status[m.status]}")

    return sol

# plot della soluzione
def plotSol(sol,dfdata,dfpoints):
   ymin   = dfpoints.iloc[:,1].min()
   ymax   = dfpoints.iloc[:,1].max()
   yrange = ymax-ymin

   fig, ax = plt.subplots(figsize=(10,6))
   ax.plot(dfpoints.iloc[:,1],marker='.',linewidth=0,color='blue')
   for i in sol:
      t1, t2, m, q = dfdata.loc[i,'low'], dfdata.loc[i,'hi'], dfdata.loc[i,'m'], dfdata.loc[i,'q']
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
   # ---------------------- data reading section
   dirpath   = "/Users/lisavecchi/Desktop/segmentation/data/M3/N1879.csv"
   runspath  = "/Users/lisavecchi/Desktop/segmentation/data/M3/N1879_runs.csv" # file con tutti i segmenti
   name      = runspath.split('/')[-1].rsplit('.', 1)[0]
   dataapath = dirpath + runspath
   df = pd.read_csv(dataapath,usecols=[1,2,3,4,5])   # i segmenti fra cui scegliere
   points    = runspath.replace("_runs", "")
   dfpoints  = pd.read_csv(dirpath+points)           # i punti da interpolare
   
   # ---------------------- solution section
   low  = df.loc[:,'low'].values
   high = df.loc[:,'hi'].values
   cost = df.loc[:,'cost'].values
   maxNseg = 4
   tstart  = time.time()
   sol     = run_SPP(name,low,high,cost,maxNseg)
   tend = time.time()
   tcpu = tend-tstart
   
   # ----------------------- results output section
   plotSol(sol,df,dfpoints)
   print(f'fine, t.cpu = {tcpu:.2f}')
