#include "cuttingPlanes.h"
#include "cplexMIP.h"

#include <ilcplex/cplex.h>
#include <vector>
#include <tuple>
#include <string>
#include <sstream>
#include <iostream>
#include <ctime>

using namespace std;

// separa tagli del tipo segnmento su segmento giu
vector<int> separateUpDownCuts(const vector<double>& x,
   vector<tuple<int, int, double, double, double>> lstOLS)
{  int i,j, endi, startj;
   double m1,m2,eps = 0.0001;
   vector<int> newCut;
   bool isCut = false;
   for(i=0;i<x.size();i++)
      if(x[i]>eps)
      {  cout << "Trying cut on " << i << endl;
         newCut.push_back(i);
         endi = get<1>(lstOLS[i]);
         for (j=0;j<x.size();j++)
         {  m1 = get<2>(lstOLS[i]);
            m2 = get<2>(lstOLS[j]);
            startj = get<0>(lstOLS[j]);
            if(startj == endi+1)
               if (m1*m2>0)
               {  newCut.push_back(j);
                  if(x[j] > 0.001) 
                  {  isCut = true;
                     cout << "clash on " << j << endl;
                  }
               }
         }
         if(isCut) break;
         newCut.clear();
      }
   if(!isCut) newCut.clear();
   cout << "New cut: ";
   for(i=0;i<newCut.size();i++) cout << newCut[i] << " ";
   cout << endl;
   return newCut;
}

// Adds one cut of the form  x_k1 + x_k2 + ... <= 1 where the indices are in newCoef
int addConflictCut(CPXENVptr env, CPXLPptr lp, const vector<int>& newCoef, int cutId)
{  int status = 0;

   if (newCoef.empty()) return 0;

   int cur_numcols = CPXgetnumcols(env, lp);
   if (cur_numcols < 0)
   {  cout << "ERROR: could not get number of columns." << endl;
      return 1;
   }

   // Check indices
   for (int h = 0; h < (int)newCoef.size(); h++)
   {
      if (newCoef[h] < 0 || newCoef[h] >= cur_numcols)
      {  cout << "ERROR: invalid column index in cut: " << newCoef[h] << endl;
         return 1;
      }
   }

   int rcnt = 1;                     // one row
   int nzcnt = (int)newCoef.size();  // one nonzero per listed variable

   double rhs[1];
   char sense[1];
   int rmatbeg[1];

   rhs[0] = 1.0;
   sense[0] = 'L';
   rmatbeg[0] = 0;

   vector<int> rmatind;
   vector<double> rmatval;
   vector<string> rowname;

   for (int h = 0; h < (int)newCoef.size(); h++)
   {  rmatind.push_back(newCoef[h]);
      rmatval.push_back(1.0);
   }

   ostringstream osstr;
   osstr << "cut_" << cutId;
   rowname.push_back(osstr.str());

   char** rname = new char*[1];
   rname[0] = const_cast<char*>(rowname[0].c_str());

   status = CPXaddrows(env, lp,
      0,                 // no new columns
      rcnt,              // 1 new row
      nzcnt,             // number of nonzeros
      rhs,
      sense,
      rmatbeg,
      &rmatind[0],
      &rmatval[0],
      NULL,
      rname);

   delete[] rname;

   if (status)
      cout << "ERROR: CPXaddrows failed when adding cut " << cutId << endl;

   return status;
}

// Cplex, to populate by row, we first create the columns, and then add the rows.
int populatebyrowCPX(CPXENVptr env, CPXLPptr lp, vector<double> y, 
   vector<tuple<int, int, double, double, double>> lstOLS, int nMaxSegm)
{
   int status, numrows, numcols, numnz, i, j, n, m;
   vector<double> obj;
   vector<double> lb;
   vector<double> ub;
   vector<string> colname;
   vector<int>    rmatbeg;
   vector<int>    rmatind;
   vector<double> rmatval;
   vector<double> rhs;
   vector<char>   sense;
   vector<string> rowname;

   status = numrows = numcols = numnz = 0;

   n = (int)lstOLS.size();
   m = (int)y.size();

   status = CPXchgobjsen(env, lp, CPX_MIN);  // minimization
   if (status) cout << "ERROR" << endl;

   // Create columns
   for (j = 0; j < n; j++)
   {  obj.push_back(get<4>(lstOLS[j]));
      lb.push_back(0.0);
      ub.push_back(1.0);

      ostringstream osstr;
      osstr << "x" << j;
      colname.push_back(osstr.str());
      numcols++;
   }

   char** cname = new char*[colname.size()];
   for (int index = 0; index < (int)colname.size(); index++)
      cname[index] = const_cast<char*>(colname[index].c_str());

   status = CPXnewcols(env, lp, numcols, &obj[0], &lb[0], &ub[0], NULL, cname);
   delete[] cname;
   if (status) cout << "ERROR" << endl;

   // Covering / partitioning constraints
   for (i = 0; i < m; i++)
   {
      rmatbeg.push_back(numnz);
      numrows++;

      ostringstream osstr;
      osstr << "c" << i;
      rowname.push_back(osstr.str());

      for (j = 0; j < n; j++)
      {
         if (i >= get<0>(lstOLS[j]) && i <= get<1>(lstOLS[j]))
         {
            rmatind.push_back(j);
            rmatval.push_back(1.0);
            numnz++;
         }
      }

      sense.push_back('E');
      rhs.push_back(1.0);

      if (i % 10 == 0 || i == m - 1)
         cout << "Constr " << i << endl;
   }

   // Max cardinality constraint
   rmatbeg.push_back(numnz);
   numrows++;
   {
      ostringstream osstr;
      osstr << "maxn";
      rowname.push_back(osstr.str());
   }

   for (j = 0; j < n; j++)
   {
      rmatind.push_back(j);
      rmatval.push_back(1.0);
      numnz++;
   }

   sense.push_back('L');
   rhs.push_back(nMaxSegm);

   cout << "Constr max num" << endl;

   // Min cardinality constraint
   int nMinSegm = 6;
   rmatbeg.push_back(numnz);
   numrows++;
   {
      ostringstream osstr;
      osstr << "minn";
      rowname.push_back(osstr.str());
   }

   for (j = 0; j < n; j++)
   {
      rmatind.push_back(j);
      rmatval.push_back(1.0);
      numnz++;
   }

   sense.push_back('G');
   rhs.push_back(nMinSegm);

   cout << "Constr minn" << endl;

   char** rname = new char*[rowname.size()];
   for (int index = 0; index < (int)rowname.size(); index++)
      rname[index] = const_cast<char*>(rowname[index].c_str());

   status = CPXaddrows(env, lp, 0, numrows, numnz,
      &rhs[0], &sense[0], &rmatbeg[0],
      &rmatind[0], &rmatval[0], NULL, rname);

   delete[] rname;

   return status;
}

// Main cutting-plane / outer branch-and-cut style routine
vector<double> goCutPlanes(vector<double> y,
   vector<tuple<int, int, double, double, double>> lstOLS,
   int ntot, string cons)
{
   int i,cont;
   int cur_numrows = -1, cur_numcols = -1;
   int status = 0;
   int solstat = 0;
   double objval = -1.0, tCpuOpt = 0.0;
   clock_t tstart, tLPend, tMIPend;

   CPXENVptr env = NULL;
   CPXLPptr  lp  = NULL;

   vector<double> x;
   vector<char> ctype;

   int n = (int)lstOLS.size();
   int cutId = 0;

   // Open CPLEX
   env = CPXopenCPLEX(&status);
   if (env == NULL)
   {  char errmsg[CPXMESSAGEBUFSIZE];
      cout << "Could not open CPLEX environment." << endl;
      CPXgeterrorstring(env, status, errmsg);
      cout << errmsg << endl;
      goto TERMINATE;
   }

   status = CPXsetintparam(env, CPXPARAM_ScreenOutput, CPX_OFF);
   if (status)
   {  cout << "Failure to turn on screen indicator, error " << status << endl;
      goto TERMINATE;
   }

   status = CPXsetintparam(env, CPXPARAM_Read_DataCheck, CPX_DATACHECK_WARN);
   if (status)
   {  cout << "Failure to turn on data checking, error " << status << endl;
      goto TERMINATE;
   }

   // Create MIP problem
   lp = CPXcreateprob(env, &status, "linSegm");
   if (lp == NULL)
   {  cout << "Failed to create LP." << endl;
      goto TERMINATE;
   }

   status = populatebyrowCPX(env, lp, y, lstOLS, ntot);
   if (status)
   {  cout << "Failed to populate problem." << endl;
      goto TERMINATE;
   }

   cur_numrows = CPXgetnumrows(env, lp);
   cur_numcols = CPXgetnumcols(env, lp);

   cout << "Initial model: rows " << cur_numrows << " cols " << cur_numcols << endl;

   x.assign(cur_numcols, 0.0);
   tstart = clock();

   // PHASE 1: LP CUT GENERATION LOOP
   cout << "----- START LP CUT GENERATION -----" << endl;
   cont = 0;
   while (cont < 0)
   {  status = CPXlpopt(env, lp);
      if (status)
      {  cout << "Failed to optimize LP relaxation." << endl;
         goto TERMINATE;
      }

      status = CPXgetx(env, lp, &x[0], 0, cur_numcols - 1);
      if (status)
      {  cout << "Failed to get LP solution x." << endl;
         goto TERMINATE;
      }

      vector<int> newCoef;
      if(cons=="updown")
      {
         newCoef = separateUpDownCuts(x,lstOLS);
         if (newCoef.empty())
         {  cout<<"No more violated cuts found at LP level."<<endl;
            break;
         }

         status = addConflictCut(env, lp, newCoef, cutId);
         if (status)
         {  cout<<"Failed to add LP cut "<<cutId<<endl;
            goto TERMINATE;
         }
         cout<<"Added LP cut "<<cutId<<" with "<<newCoef.size()<<" variables."<<endl;
      }
      cutId++;
      cur_numrows = CPXgetnumrows(env, lp);
      cont++;
   }

   tLPend = clock();
   cout << "LP cut generation time: "
      << (double)(tLPend - tstart) / CLOCKS_PER_SEC << endl;

   // Set all variables to binary
   ctype.clear();
   for (i = 0; i < cur_numcols; i++)
      ctype.push_back('B');

   status = CPXcopyctype(env, lp, &ctype[0]);
   if (status)
   {  cout << "Failed to restore ctype before MIP phase." << endl;
      goto TERMINATE;
   }

   // PHASE 2: MIP CUT GENERATION LOOP
   cout << "----- START MIP CUT GENERATION -----" << endl;
   while (true)
   {
      status = CPXmipopt(env, lp);
      if (status)
      {  cout << "Failed to optimize MIP." << endl;
         goto TERMINATE;
      }

      solstat = CPXgetstat(env, lp);
      cout << "MIP status = " << solstat << endl;

      status = CPXgetobjval(env, lp, &objval);
      if (status)
      {  cout << "No MIP objective value available. Exiting..." << endl;
         goto TERMINATE;
      }

      cout << "Current MIP objective value = " << objval << endl;

      status = CPXgetx(env, lp, &x[0], 0, cur_numcols - 1);
      if (status)
      {  cout << "Failed to get MIP solution x." << endl;
         goto TERMINATE;
      }
      for(i=0;i<x.size();i++)
         if(x[i] > 0.001) cout << i << " ";
      cout << endl;

      if(cons == "updown")
      {  vector<int> newCoef = separateUpDownCuts(x,lstOLS);

         if (newCoef.empty())
         {  cout<<"No more violated cuts found at MIP level."<<endl;
            break;
         }

         status = addConflictCut(env, lp, newCoef, cutId);
         if (status)
         {  cout<<"Failed to add MIP cut "<<cutId<<endl;
            goto TERMINATE;
         }

         cout<<"Added MIP cut "<<cutId<<" with "<<newCoef.size()<<" variables."<<endl;

         cutId++;
         cont++;
         cur_numrows = CPXgetnumrows(env, lp);
      }
   }

   tMIPend = clock();
   tCpuOpt = (double)(tMIPend - tstart) / CLOCKS_PER_SEC;

   // Final solution info
   solstat = CPXgetstat(env, lp);
   cout << "Final solution status = " << solstat << endl;

   status = CPXgetobjval(env, lp, &objval);
   if (status)
   {  cout << "No final objective value available." << endl;
      goto TERMINATE;
   }

   cout << "Final solution value = " << objval << " num cuts " << cont << endl;
   cout << "Total CPU time = " << tCpuOpt << endl;

   status = CPXgetx(env, lp, &x[0], 0, cur_numcols - 1);
   if (status)
   {  cout << "Failed to get final x." << endl;
      goto TERMINATE;
   }

   if (cur_numcols < 200)
   {  status = CPXwriteprob(env, lp, "problem_final.lp", NULL);
      if (status)
         cout << "Failed to write model to disk." << endl;
   }

TERMINATE:
   if (lp != NULL)
   {  status = CPXfreeprob(env, &lp);
      if (status)
         cout << "CPXfreeprob failed, error code " << status << endl;
   }

   if (env != NULL)
   {  status = CPXcloseCPLEX(&env);
      if (status)
      {  char errmsg[CPXMESSAGEBUFSIZE];
         cout << "Could not close CPLEX environment." << endl;
         CPXgeterrorstring(env, status, errmsg);
         cout << errmsg << endl;
      }
   }

   return x;
}
