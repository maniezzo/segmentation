// SC_CDP risolto con gurobi
// API Mapping (CPLEX -> Gurobi)
// CPXopenCPLEX   -> GRBEnv env; GRBModel model(env);
// CPXchgobjsen   -> model.set(GRB_IntAttr_ModelSense, 1 /* Minimize */); or model.setObjective(expr, GRB_MINIMIZE)
// CPXnewcols     -> model.addVar(lb, ub, objcoef, GRB_CONTINUOUS, name)
// CPXaddrows     -> model.addConstr(sum >= rhs, name)
// CPXlpopt       -> model.optimize() with continuous vars
// CPXsolution    -> pull attributes: X, RC, Pi, Slack
// CPXcopyctype   -> set each var’s VType to GRB_BINARY
// CPXmipopt      -> model.optimize() after setting integrality
// CPXwriteprob   -> model.write("problem.lp")
// 
// Build with MSVC, add the Gurobi include and link against the Gurobi C++ library (gurobi_c++.lib) in your MSVC project, 
// and ensure the Gurobi DLL is in your PATH at runtime.
// Make sure to include gurobi_c++.h and set library paths accordingly.
#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <ctime>
#include "gurobiMIP.h"

// Populate model with variables and covering constraints
int populateGurobiByRow(GRBModel& model,
   const vector<double>& y,
   const vector<tuple<int,int,double,double,double>>& lstOLS,
   vector<GRBVar>& xVars,
   vector<GRBConstr>& constrs)
{
   try 
   {  int n = (int)lstOLS.size();
      int m = (int)y.size();

      constrs.reserve(m);
      model.set(GRB_IntAttr_ModelSense, 1); // Minimize

      xVars.resize(n);
      for (int j = 0; j < n; ++j) 
      {  double obj = get<4>(lstOLS[j]);
         string name = "x" + to_string(j);
         xVars[j] = model.addVar(0.0, 1.0, obj, GRB_CONTINUOUS, name);
      }
      model.update();

      for (int i = 0; i < m; ++i) 
      {  GRBLinExpr sum = 0.0;  // Ensure this is GRBLinExpr
         for (int j = 0; j < n; ++j) 
         {  int start = get<0>(lstOLS[j]);
            int end   = get<1>(lstOLS[j]);
            if (i >= start && i <= end) 
               sum += xVars[j];
         }
         string cname = "c" + to_string(i);
         GRBConstr c = model.addConstr(sum == 1.0, cname);  // <<<<<<<<<<<<<<<<<<< PARTITIONING OPPURE COVERING == oppure <=
         constrs.push_back(c);
      }
      model.update();
      return 0;
   } 
   catch (GRBException& e) 
   {  cerr << "Gurobi Error: " << e.getErrorCode() << " " << e.getMessage() << endl;
      return e.getErrorCode();
   } 
   catch (...) 
   {  cerr << "Unknown error in populateByRow" << endl;
      return -1;
   }
}

// Main function
vector<double> goGurobi(vector<double> y, vector<tuple<int, int, double, double, double>> lstOLS) 
{
   clock_t tstart, truns, tMIP;
   vector<double> xnil;

   try 
   {  
      int n = lstOLS.size();
      // Create environment and model
      GRBEnv env = GRBEnv(true);
      env.set(GRB_IntParam_OutputFlag, 1);
      env.start();

      GRBModel model(env);
      model.set(GRB_StringAttr_ModelName, "ExtendedSetCover");

      vector<GRBVar> xVars;
      vector<GRBConstr> constrs;

      int status = populateGurobiByRow(model, y, lstOLS, xVars, constrs);
      if (status) 
      {  cout << "Failed to populate model." << endl;
         goto TERMINATE;
      }

      // ---------- Solve LP ----------
      model.optimize();

      int numRows = model.get(GRB_IntAttr_NumConstrs);
      int numCols = model.get(GRB_IntAttr_NumVars);
      cout << "LP: rows=" << numRows << " cols=" << numCols << endl;

      double objval = model.get(GRB_DoubleAttr_ObjVal);
      cout << "LP objective = " << objval << endl;

      // Retrieve LP solution
      vector<double> x(numCols), rc(numCols), pi(numRows), slack(numRows);
      for (int j = 0; j < numCols; ++j) 
      {  x[j]  = xVars[j].get(GRB_DoubleAttr_X);
         rc[j] = xVars[j].get(GRB_DoubleAttr_RC);
      }
      for (int i = 0; i < numRows; ++i) 
      {  pi[i]    = constrs[i].get(GRB_DoubleAttr_Pi);
         slack[i] = constrs[i].get(GRB_DoubleAttr_Slack);
      }

      // ---------- Switch to MIP ----------
      for (int j = 0; j < numCols; ++j)
         xVars[j].set(GRB_CharAttr_VType, GRB_BINARY);
      model.update();

      clock_t tStart = clock();
      model.optimize();
      clock_t tEnd = clock();
      cout << "MIP solve time: " << (double)(tEnd - tStart) / CLOCKS_PER_SEC << " sec" << endl;

      int solstat = model.get(GRB_IntAttr_Status);
      cout << "Solution status = " << solstat << endl;

      // redefine to hold the integer solution
      x.clear();
      if (solstat == GRB_OPTIMAL) 
      {  objval = model.get(GRB_DoubleAttr_ObjVal);
         cout << "MIP objective = " << objval << endl;
         for (int j = 0; j < numCols; ++j)
         {  x.push_back(xVars[j].get(GRB_DoubleAttr_X));
            //cout<<"x["<<j<<"] = "<<x[j]<<endl;
         }
      }

      if (numCols < 200)
         model.write("problem.lp");
      return x;
   } 
   catch (GRBException& e) 
   {  cout << "Gurobi exception: " << e.getErrorCode() << " " << e.getMessage() << endl;
      goto TERMINATE;
   } 
   catch (...) 
   {  cout << "Unknown exception." << endl;
      goto TERMINATE;
   }

TERMINATE: return xnil;
}
