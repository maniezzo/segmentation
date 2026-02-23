#include "cplexMIP.h"

// Cplex, to populate by row, we first create the columns, and then add the rows.
int populatebyrow(CPXENVptr env, CPXLPptr lp, vector<double> y, vector<tuple<int, int, double, double, double>> lstOLS, int ntot)
{  int status,numrows,numcols,numnz,i,j,n,m;
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

   n = lstOLS.size();
   m = y.size();

   status = CPXchgobjsen(env, lp, CPX_MIN);  // Problem is minimization
   if (status) cout << "ERROR" << endl;

   // Now create the new columns.
   for(j=0;j<n;j++)
   {  obj.push_back(get<4>(lstOLS[j]));
      lb.push_back(0.0);
      ub.push_back(1.0);
      ostringstream osstr;
      osstr << "x" << j;
      colname.push_back(osstr.str());
      numcols++;
   }

   // vector<string> to char**
   char** cname = new char* [colname.size()];
   for (int index = 0; index < colname.size(); index++)
      cname[index] = const_cast<char*>(colname[index].c_str());

   status = CPXnewcols(env, lp, numcols, &obj[0], &lb[0], &ub[0], NULL, cname);
   delete[] cname;
   if (status)  cout << "ERROR" << endl;

   // The covering constraints
   for (i=0;i<m;i++)
   {
      rmatbeg.push_back(numnz); numrows++;
      ostringstream osstr;
      osstr << "c" << i;
      rowname.push_back(osstr.str());
      for (j=0;j<n;j++)
         if (i >= get<0>(lstOLS[j]) && i <= get<1>(lstOLS[j]))
         {
            rmatind.push_back(j); 
            rmatval.push_back(1.0); 
            numnz++;
         }
      sense.push_back('E');
      rhs.push_back(1.0);
      if(i%10 == 0 || i==m-1)
         cout << "Constr" << i << endl;
   }

   // maxnum constraint
   rmatbeg.push_back(numnz); numrows++;
   ostringstream osstr;
   osstr << "ntot";
   rowname.push_back(osstr.str());
   for (j=0;j<n;j++)
   {
      rmatind.push_back(j); 
      rmatval.push_back(1.0); 
      numnz++;
   }
   sense.push_back('L');
   rhs.push_back(ntot);
   cout << "Constr ntot"<< endl;

   // vector<string> to char**
   char** rname = new char* [rowname.size()];
   for (int index = 0; index < rowname.size(); index++) {
      rname[index] = const_cast<char*>(rowname[index].c_str());
   }
   status = CPXaddrows(env, lp, 0, numrows, numnz, &rhs[0], &sense[0], &rmatbeg[0], &rmatind[0], &rmatval[0], NULL, rname);
   delete[] rname;
   if (status)  goto TERMINATE;

   TERMINATE:
   return (status);
} 


vector<double> goCPLEX(vector<double> y, vector<tuple<int, int, double, double, double>> lstOLS, int ntot)
{  int i,j;
   int       cur_numrows=-1, cur_numcols=-1;
   int       status = 0;
   CPXENVptr env = NULL;
   CPXLPptr  lp = NULL;
   int       solstat, n_brk=-1;
   double    objval=-1, tCpuOpt, cost = 0;
   clock_t   tstart, truns, tMIP;

   vector<double> x;
   vector<double> pi;
   vector<double> slack;
   vector<double> dj;
   vector<char>   ctype;

   int n = lstOLS.size();
   // Initialize the CPLEX environment
   env = CPXopenCPLEX(&status);
   if (env == NULL)
   {  char  errmsg[CPXMESSAGEBUFSIZE];
      cout << "Could not open CPLEX environment." << endl;
      CPXgeterrorstring(env, status, errmsg);
      cout << errmsg << endl;
      goto TERMINATE;
   }

   // Turn on output to the screen 
   status = CPXsetintparam(env, CPXPARAM_ScreenOutput, CPX_ON);
   if (status)
   {  cout << "Failure to turn on screen indicator, error " << status << endl;
      goto TERMINATE;
   }

   // Turn on data checking
   status = CPXsetintparam(env, CPXPARAM_Read_DataCheck, CPX_DATACHECK_WARN);
   if (status)
   {  cout << "Failure to turn on data checking, error " << status << endl;
      goto TERMINATE;
   }

   // Create the problem.
   lp = CPXcreateprob(env, &status, "linSegm");
   if (lp == NULL)
   {  cout << "Failed to create LP." << endl;
      goto TERMINATE;
   }

   // Now populate the problem with the data.
   status = populatebyrow(env, lp, y, lstOLS, ntot);
   if (status)
   {  cout << "Failed to populate problem." << endl;
      goto TERMINATE;
   }

   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   tstart = clock();
   status = CPXlpopt(env, lp);
   if (status)
   {  cout << "Failed to optimize LP." << endl;
      goto TERMINATE;
   }

   cur_numrows = CPXgetnumrows(env, lp);
   cur_numcols = CPXgetnumcols(env, lp);
   cout << "num rows " << cur_numrows << " num cols " << cur_numcols << endl;

   for (int j = 0; j < cur_numcols; j++)
   {  x.push_back(0);  // primal values
      dj.push_back(0); // reduced costs
   }

   for (int i = 0; i < cur_numrows; i++)
   {  pi.push_back(0);     // dual values
      slack.push_back(0);  // constraint slacks
   }

   status = CPXsolution(env, lp, &solstat, &objval, &x[0], &pi[0], &slack[0], &dj[0]);
   if (status)
   {  cout << "Failed to obtain solution." << endl;
      goto TERMINATE;
   }

   // Write the output to the screen.
   //cout << "\nSolution status = " << solstat << endl;
   //cout << "Solution value  = "   << objval << endl;
   //for (i = 0; i < cur_numrows; i++) 
   //   cout << "Row "<< i << ":  Slack = "<< slack[i] <<"  Pi = " << pi[i] << endl;

   //for (j = 0; j < cur_numcols; j++) 
   //   cout << "Column " << j << ":  Value = " << x[j] <<"  Reduced cost = " << dj[j] << endl;

   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> MIP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   for (i = 0; i < n; i++)
      ctype.push_back('I');
   status = CPXcopyctype(env, lp, &ctype[0]);
   if (status)
   {  cout << "Failed to copy ctype" << endl;
      goto TERMINATE;
   }

   // ---------------------------- Optimize to integrality
   status = CPXmipopt(env, lp);
   if (status)
   {  cout << "Failed to optimize MIP" << endl;
      goto TERMINATE;
   }

   tMIP = clock();
   tCpuOpt = (tMIP - tstart) / CLOCKS_PER_SEC;
   cout << "CPU time for MIP: " << tCpuOpt << endl;

   solstat = CPXgetstat(env, lp);
   cout << "Solution status = " << solstat << endl;

   status = CPXgetobjval(env, lp, &objval);
   if (status)
   {  cout << "No MIP objective value available.  Exiting..." << endl;
   goto TERMINATE;
   }

   cout << "Solution value  = " << objval << endl;
   cur_numrows = CPXgetnumrows(env, lp);
   cur_numcols = CPXgetnumcols(env, lp);

   status = CPXgetx(env, lp, &x[0], 0, cur_numcols - 1);
   if (status)
   {  cout << "Failed to get optimal integer x." << endl;
      goto TERMINATE;
   }

   // Finally, write a copy of the problem to a file
   if (cur_numcols < 200)
   {
      status = CPXwriteprob(env, lp, "problem.lp", NULL);
      if (status)
         cout << "Failed to write model to disk." << endl;
   }

TERMINATE:
   // Free up the problem as allocated by CPXcreateprob, if necessary
   if (lp != NULL)
   {
      status = CPXfreeprob(env, &lp);
      if (status)
         cout << "CPXfreeprob failed, error code " << status << endl;
   }

   // Free up the CPLEX environment, if necessary
   if (env != NULL)
   {  status = CPXcloseCPLEX(&env);
      if (status)
      {  char  errmsg[CPXMESSAGEBUFSIZE];
         cout << "Could not close CPLEX environment." << endl;
         CPXgeterrorstring(env, status, errmsg);
         cout << errmsg << endl;
      }
   }
   return (x);
}