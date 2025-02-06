import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
#from ortools.linear_solver import pywraplp
import pulp
import scipy.stats
from scipy.stats import gamma, norm, nbinom
import nelder_mead as nm
from PSO import PSO
import chow_test

def covid_nbinom(pars):
   return pars[2] * nbinom.pmf(xval, pars[0], pars[1])

def loss_nbinom(pars):
   return ((y - covid_nbinom(pars)) ** 2).replace(np.nan, np.inf).sum()

def fit_nbinom(xval):
   PSOopt = PSO(nump=50, objFunc=loss_nbinom, ndim=3, maxiter=500, numneigh=5,
                x_min=np.array([10, 0.1, 109000.]), x_max=np.array([1000, 100, 10000000.]),
                isMax=False, c0=0.25, c1=2, c2=2, isVerbose=True)
   # x_min = np.array([0.1, 0.1, 50000.]), x_max = np.array([0.5, 0.5, 200000.]),
   res = PSOopt.runPSO()
   nbinom_best = nm.nelder_mead(loss_nbinom, res)
   print(nbinom_best)
   nbinom_rmse = np.sqrt(nbinom_best[1] / len(xval))
   return nbinom_best, nbinom_rmse

def pulpModel(y,lstOLS):
   solver_list = pulp.listSolvers(onlyAvailable=True)
   solver = pulp.getSolver('CPLEX_CMD', timeLimit=500)
   n = len(lstOLS)
   m = len(y)

   # Create the 'prob' variable to contain the problem data
   prob = pulp.LpProblem("Linear segmentation", pulp.LpMinimize)

   # Decision variables
   x = pulp.LpVariable.dicts('x', range(n), lowBound=0, upBound=1, cat="Integer")
   print("LP: number of variables =", len(x))

   # The objective function
   prob += pulp.lpSum([x[j]*lstOLS[j][4] for j in range(n)]), "cost"

   # The covering constraints
   for i in range(m):
      coeff = np.zeros(n)
      for j in range(n):
         if (i >= lstOLS[j][0] and i <= lstOLS[j][1]):
            coeff[j]=1
      sumreq = pulp.lpSum([x[j] * coeff[j] for j in range(n)])
      prob += (sumreq >= 1, "Cnstr{}".format(i))
      print(f"Constr{i}")

   # The problem data is written to an .lp file
   if(n<500):
      prob.writeLP("linearSegmentation.lp")

   # prob.solve()
   res = prob.solve(solver)
   #prob.solve(pulp.PULP_CBC_CMD(gapRel=0.00001, timeLimit=500, threads=None))

   lstSol = []
   if pulp.LpStatus[prob.status] == "Optimal":
      print("Total cost {0} Chosen sites {1}:".format(pulp.value(prob.objective), len(x)))
      for i in x:
         if x[i].varValue > 0.001:
            print("x_{0} value {1}".format(i, x[i].varValue))
            lstSol.append(i)
   return lstSol

# chi ^ 2
def cost(low,up,y):
   x = low + np.arange(len(y))
   A = np.vstack([x, np.ones(len(x))]).T
   m, q = np.linalg.lstsq(A, y, rcond=None)[0]

   ypred = m * x + q
   residuals = y - ypred

   # Compute chi-square
   variance = np.var(y)
   chi_square = 0
   if(variance>0):
      chi_square = np.sum(residuals ** 2) / variance
   return (low,up,m,q,chi_square)

def costR2(low,up,y):
   m, q, r_value, p_value, std_err = scipy.stats.linregress(range(len(y)), y)
   r2cost = 1-r_value
   return (low,up,m,q,r2cost)

def costMSE(low,up,y):
   x = low + np.arange(len(y))
   A = np.vstack([x, np.ones(len(x))]).T
   m, q = np.linalg.lstsq(A, y, rcond=None)[0]
   ypred = m * x + q
   MSEcost = np.square(np.subtract(y, ypred)).mean()
   return (low,up,m,q,MSEcost)

def costSER(low,up,y):
   x = low + np.arange(len(y))
   A = np.vstack([x, np.ones(len(x))]).T
   m, q = np.linalg.lstsq(A, y, rcond=None)[0]
   ypred = m * x + q
   residuals = y - ypred
   variance = np.var(residuals)
   SERcost = np.sqrt(variance/len(residuals))
   return (low,up,m,q,SERcost)

def costNegBin(low,up,y):
   xval = low + np.arange(len(y))
   nbinom_best,nbinom_rmse = fit_nbinom(xval)
   NBcost = nbinom_rmse
   return (low,up,nbinom_best[0][0],nbinom_best[0][1],nbinom_best[0][2],NBcost)

def linprob(y,lstOLS):
   # Create the linear solver
   solver = pywraplp.Solver.CreateSolver("GLOP")
   if not solver: return
   # Create the variables
   x = {}
   for j in range(len(lstOLS)):
      x[j] = solver.NumVar(0, 1, "x[%i]" % j)
   print("LP: number of variables =", solver.NumVariables())
   # Create constraints
   constraints = []
   for i in range(len(y)):
      constraints.append(solver.Constraint(1, 1,f"c{i}"))
      for j in range(len(lstOLS)):
         if(i>=lstOLS[j][0] and i<=lstOLS[j][1]):
            constraints[i].SetCoefficient(x[j], 1)
   print("LP: number of constraints =", solver.NumConstraints())
   # Create the objective function
   objective = solver.Objective()
   for j in range(len(lstOLS)):
      objective.SetCoefficient(x[j], lstOLS[j][4])
   objective.SetMinimization()

   if(len(x) < 30):
      with open("problem.lp", "w") as fout:
         s = solver.ExportModelAsLpFormat(False)
         fout.write(s)

   status = solver.Solve()

   lstSol = []
   if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
      print(f"obj = {solver.Objective().Value()}")
      for j in range(len(x)):
         if(x[j].solution_value()>0.001):
            print(f"x{j} = {x[j].solution_value()}, reduced cost = {x[j].reduced_cost()}")
            lstSol.append(j)
      '''
      for i in range(len(constraints)):
         print(f"constr dual value = {constraints[i].dual_value()}")
      '''
   return lstSol

def intprob(y,lstOLS):
   # Create the integer solver
   solver = pywraplp.Solver.CreateSolver("CPLEX")
   if not solver: return
   # Create the variables
   x = {}
   for j in range(len(lstOLS)):
      x[j] = solver.IntVar(0, 1, "x[%i]" % j)
   print("\nMIP, number of variables =", solver.NumVariables())
   # Create constraints
   constraints = []
   for i in range(len(y)):
      constraints.append(solver.Constraint(1, 1,f"c{i}"))
      for j in range(len(lstOLS)):
         if(i>=lstOLS[j][0] and i<=lstOLS[j][1]):
            constraints[i].SetCoefficient(x[j], 1)
   print("MIP: number of constraints =", solver.NumConstraints())
   # Create the objective function
   objective = solver.Objective()
   for j in range(len(lstOLS)):
      objective.SetCoefficient(x[j], lstOLS[j][4])
   objective.SetMinimization()

   # Display output
   solver.EnableOutput()

   #print(solver.ExportModelAsLpFormat(False).replace('\\', '').replace(',_', ','), sep='\n')

   status = solver.Solve()

   lstSol = []
   if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
      print(f"obj = {solver.Objective().Value()}")
      for j in range(solver.NumVariables()):
         if(x[j].solution_value()>0.001):
            print(f"x{j} = {x[j].solution_value()}")
            lstSol.append(j)
      print("Problem solved in %f milliseconds" % solver.wall_time())
      print("Problem solved in %d iterations" % solver.iterations())
      print("Problem solved in %d branch-and-bound nodes" % solver.nodes())

   return lstSol

if __name__ == "__main__":
   dataset = "test"
   df = pd.read_csv(f"..//..//data//home//{dataset}.csv");   #"BTC-USD.csv");  #"test.csv"); "Covid_italia_22"
   y = df.iloc[:,1]#.values[:100]
   x = np.arange(len(y))
   xval = np.arange(len(y))
   minlag = max(int(len(y)/50),10);
   if (dataset == "Covid_italia_22"): minlag = 40

   lstOLS = []
   for low in np.arange(len(x)-minlag+1):
      for up in np.arange(low+minlag-1,len(x)):
         if(dataset == "Covid_italia_22"):
            tup = costNegBin(low,up,y[low:up+1])
            print(f"low {low} up {up}")
         else:
            #tup = cost(low,up,y[low:up+1]) # low,up,m,q,chi_square, low and up included
            #tup = costR2(low,up,y[low:up+1])
            #tup = costMSE(low,up,y[low:up+1])
            tup = costSER(low,up,y[low:up+1])
         lstOLS.append(tup)
      if (low % 10 == 0): print(f">>>>>>>>>>>>>>>>>>>>>>>>>>> low = {low}")

   import csv
   fields = ["uno","dos","tres","cuatro","cinque"]
   with open('runs.csv', 'w') as f:
      write = csv.writer(f)
      write.writerow(fields)
      write.writerows(lstOLS)

   solver = "CPLEX"
   if solver == "CPLEX":
      lstSol = pulpModel(y,lstOLS)
   else:
      lstSol = linprob(y,lstOLS)
      lstSol = intprob(y,lstOLS)

   lines = []
   for i in range(len(lstSol)):
      j = lstSol[i]
      m = lstOLS[j][2]
      q = lstOLS[j][3]
      x1 = lstOLS[j][0]
      y1 = m*x1+q
      x2 = lstOLS[j][1]
      y2 = m*x2+q
      segm = [(x1,y1),(x2,y2)]
      lines.append(segm)
   lc = mpl.collections.LineCollection(lines, linewidths=2, color = 'r', label = "OLS segments")

   fig, ax = plt.subplots()
   ax.plot(x, y, 'o', label='Original data', markersize=3)
   ax.add_collection(lc)
   ax.autoscale()
   ax.margins(0.1)
   plt.legend()
   plt.show()
   pass