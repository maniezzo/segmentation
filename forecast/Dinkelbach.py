import gurobipy as gp
from gurobipy import GRB
import pandas as pd

# Initialise lambda with the cost of any single feasible solution.
# First solution: the standard (sum-cost) SPP.
def build_and_solve(ncol, nrow, matcons, lmbda: float, silent: bool = True) -> tuple[float, list[int]]:
   """
   Solve  min  sum_j (c_j - lmbda) * x_j
   s.t.   sum_j matcons[i][j] * x_j = 1   for all i   (SPP)
          x_j in {0,1}
   Returns (optval, selected_indices).
   """
   with gp.Env(empty=True) as env:
      env.setParam("OutputFlag", 0)
      env.start()
      with gp.Model(env=env) as m:
         x = m.addVars(ncol, vtype=GRB.BINARY, name="x")
         
         # Partition constraints
         for i in range(nrow):
            m.addConstr(
               gp.quicksum(matcons[i][j] * x[j] for j in range(ncol)) == 1,
               name=f"cover_{i}",
            )
         
         # Parametric objective
         m.setObjective(
            gp.quicksum((costs[j] - lmbda) * x[j] for j in range(ncol)),
            GRB.MINIMIZE,
         )
         m.optimize()
         
         if m.Status == GRB.OPTIMAL:
            selected = [j for j in range(ncol) if x[j].X > 0.5]
            return m.ObjVal, selected
         elif m.Status == GRB.INFEASIBLE:
            return None, None
         else:
            return None, None


# Solve a Set Partitioning Problem with average-cost objective via the Dinkelbach algorithm.
def solve_spp_average_cost(
      lstIntervals: list[tuple],
      lstCosts: list[float],
      nrows: int,
      eps: float = 1e-6,
      max_iter: int = 50,
      verbose: bool = True,
) -> dict:
   """
   Each feasible solution selects a subset S of sets such that every
   element is covered exactly once.
   The objective is to minimise the average cost of the selected sets.

       min  (1/|S|) * sum_{j in S} cost[j]

   Parameters
   ----------
   lstIntervals : list of tuples; (t1,t2))
   lstCosts     : cost[j] for set j
   nrows   : total number of elements to be covered (universe size)
   eps          : convergence tolerance on g(lambda) = min sum(c_j - lambda)*x_j
   max_iter     : maximum Dinkelbach iterations
   verbose      : print iteration log

   Returns
   -------
   dict with keys:
       'lambda_star'   : optimal average cost
       'selected_sets' : indices of sets in the optimal partition
       'iterations'    : number of Dinkelbach iterations
       'status'        : 'optimal' | 'max_iter_reached' | 'infeasible'
   """
   ncols = len(lstIntervals)
   
   # Build the constraint matrix matcons[i][j] = 1 if element i in interval (t1,t2)
   matcons = [[0] * ncols for _ in range(nrows)]
   for j, intv in enumerate(lstIntervals):
      for i in range(intv[0],intv[1]+1):
         matcons[i][j] = 1
   
   # find an initial feasible solution
   # (lambda = 0, it gives the standard min-sum SPP)
   g0, x0 = build_and_solve(ncols, nrows, matcons, lmbda=0.0)
   if x0 is None:
      return {"status": "infeasible"}
   
   lmbda = sum(lstCosts[j] for j in x0) / len(x0)   # initial lambda = F(x^0)
   
   if verbose:
      print(f"{'Iter':>5}  {'lambda':>14}  {'g(lambda)':>12}  {'|S|':>5}")
      print("-" * 44)
   
   # Dinkelbach iterations
   for k in range(max_iter):
      g_val, x_new = build_and_solve(ncols, nrows, matcons, lmbda)
      
      if x_new is None:
         return {"status": "infeasible"}
      
      lmbda_new = sum(lstCosts[j] for j in x_new) / len(x_new)
      
      if verbose:
         print(f"{k+1:>5}  {lmbda:>14.6f}  {g_val:>12.6f}  {len(x_new):>5}")
      
      if abs(g_val) < eps:          # g(lambda*) = 0  →  converged
         return {
            "lambda_star": lmbda_new,
            "selected_sets": x_new,
            "iterations": k + 1,
            "status": "optimal",
         }
      
      lmbda = lmbda_new                 # update and iterate
   
   return {
      "lambda_star": lmbda,
      "selected_sets": x_new,
      "iterations": max_iter,
      "status": "max_iter_reached",
   }

# example
if __name__ == "__main__":
   # Universe: elements {0, ... , 10}
   # 7 candidate sets
   sets = [(0,10),(0,3),(3,10),(4,8),(5,7),(8,10),(9,10)]
   costs = [5, 3, 4, 3, 4, 3, 4]
   
   result = solve_spp_average_cost(sets, costs, nrows=11, verbose=True)
   
   print()
   if result["status"] == "optimal":
      sel = result["selected_sets"]
      print(f"Status        : {result['status']}")
      print(f"Iterations    : {result['iterations']}")
      print(f"Selected sets : {sel}")
      print(f"Costs         : {[costs[j] for j in sel]}")
      print(f"Average cost  : {result['lambda_star']:.6f}")
      print(f"Check         : {sum(costs[j] for j in sel)/len(sel):.6f}")
   else:
      print(f"Status: {result['status']}")
      
   print("------------- model --------------")
   df = pd.read_csv("data/N1930models.csv")
   lstIntervals = []
   lstCosts = []
   for i in range(len(df)):
      lstIntervals.append((df.iloc[i,1],df.iloc[i,2]))
      lstCosts.append(df.iloc[i,4])
   nrows = df.iloc[-1,2]+1 # partono da 0
   result = solve_spp_average_cost(lstIntervals, lstCosts, nrows=nrows, verbose=True)
   print("fine")
        