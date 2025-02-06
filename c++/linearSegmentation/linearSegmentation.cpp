#include "linearSegmentation.h"
#include "json.h"


// checks the feasibility of a solution
bool checkFeas(vector<int> sol, vector<tuple<int, int, double, double, double>> lstOLS, double expCost)
{  int i,j,tmax,t0,t1;
   bool isFeasible = true;
   double z=0;

   if (get<0>(lstOLS[sol[0]]) != 0)
   {  cout << "-- INF, initial value not 0" << endl;
      isFeasible = false;
      goto lend;
   }

   tmax = get<1>(lstOLS[lstOLS.size()-1]);
   if (get<1>(lstOLS[sol[sol.size()-1]]) != tmax)
   {  cout << "-- INF, final value not tmax" << endl;
      isFeasible = false;
      goto lend;
   }

   t0 = 0;
   for (i = 0; i < sol.size()-1; i++)
   {  t1 = get<1>(lstOLS[sol[i]]);
      t0 = get<0>(lstOLS[sol[i+1]]); // init segment after
      z += get<4>(lstOLS[sol[i]]);
      if(t0!=t1+1)
      {  cout << "-- INF, segment do not concatenate" << endl;
         isFeasible = false;
         goto lend;
      }
   }
   z += get<4>(lstOLS[sol[i]]);
   if (z != expCost)
   {  cout << "-- INF, cost difference" << endl;
      isFeasible = false;
      goto lend;
   }

lend:    return isFeasible;
}

// single source on a DAG, min cost path beginning in tinit and ending in tend
vector<int> DAG_SSSP(int tinit, int tend, int minlag, vector<tuple<int, int, double, double, double>> lstOLS)
{  int i, j, k, n, t, currInit, maxt, maxstart;
   double c;
   vector<int> initSegm;   // indice in lstOLS inizio indici segmenti con inizio al tempo i
   vector<int> sol;

   n = lstOLS.size();
   currInit = tinit-1;

   
   for (i = 0; i < n; i++)
   {  if (get<0>(lstOLS[i]) < tinit) continue;
      if (get<0>(lstOLS[i]) > currInit)
      {  initSegm.push_back(i); // segments compatible with the time interval
         currInit = get<0>(lstOLS[i]);
      }
      if (get<0>(lstOLS[i]) > tend)
      {  initSegm.push_back(i);
         break;
      }
   }

   maxt = tend;

   vector<double> mincost(maxt+1, DBL_MAX);  // min cost for covering up to time tend
   vector<int>    minsegm(maxt+1, -1);       // id last segment producing cost mincost[t]
   maxstart = tend-minlag;

   for (t = tinit; t <= maxstart; t++)
   {  k = t - tinit;
      for (i = initSegm[k]; i < initSegm[k+1]; i++) // segments startig at time t
      {
         j = get<1>(lstOLS[i]);  // end time segment i
         c = (t > tinit ? mincost[t-1] : 0) + get<4>(lstOLS[i]);
         if (j<=tend && mincost[j] > c)
         {  mincost[j] = c;
            minsegm[j] = i;
         }
      }
   }

   sol = reconstructSolution(lstOLS, minsegm, tinit, tend);
   //cout << "(DAG) " << dsName << " cost: " << tinit << "-" << tend << ": " << std::setprecision(5) << mincost[tend] << endl;
   return sol;
}

// ricostruisce la soluzione DAG
vector<int> reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, vector<int> minsegm, int tinit, int tend)
{  int i, j, t;
   double sum = 0;
   vector<int> sol;

   t = tend;
   while (t > tinit)
   {  i = minsegm[t];
      sum += get<4>(lstOLS[i]);  // costo integrale
      sol.push_back(i);
      //cout << "Segmento " << i << " " << get<0>(lstOLS[i]) << "-" << get<1>(lstOLS[i]) << " costo " << get<4>(lstOLS[i]) << endl;
      j = get<1>(lstOLS[i]) - get<0>(lstOLS[i]) + 1;
      t -= j;
   }
   //cout << "Costo complessivo " << sum << endl;
   return sol;
}

// split di una stringa in un array di elementi delimitati da separatori
vector<string> split(string str, char sep)
{  vector<string> tokens;
   size_t start;
   size_t end = 0;
   while ((start = str.find_first_not_of(sep, end)) != std::string::npos) {
      end = str.find(sep, start);
      tokens.push_back(str.substr(start, end - start));
   }
   return tokens;
}

int readData(string dataFileName, vector<int>& X, vector<double>& Y)
{  int i, cont, id,n=0;
   double d;
   string line;
   vector<string> elem;

   // leggo i punti
   ifstream f;
   string dataSetFile = dataFileName;
   cout << "Opening datafile " << dataSetFile << endl;
   f.open(dataSetFile);
   if (f.is_open())
   {
      getline(f, line);  // headers
      cout << line << endl;
      elem = split(line, ',');

      while (getline(f, line))
      {  cont = 0;
         elem = split(line, ',');
         id = stoi(elem[0]);
         X.push_back(id);
         d = stod(elem[1]);
         Y.push_back(d);
l0:      cont++;
      }
      f.close();
      n = Y.size();  // number of input records
   }
   else cout << "Cannot open dataset input file\n";
   return n;
}

// Cplex, to populate by row, we first create the columns, and then add the rows.
int populatebyrow(CPXENVptr env, CPXLPptr lp, vector<double> y, vector<tuple<int, int, double, double, double>> lstOLS)
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
      sense.push_back('G');
      rhs.push_back(1.0);
      if(i%10 == 0 || i==m-1)
         cout << "Constr" << i << endl;
   }

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

// OLS line through vector of points
tuple<double,double> linearRegression(vector<int> x, vector<double> y)
{  int n, i;
   double sum_x = 0, sum_x2 = 0, sum_y = 0, sum_xy = 0, m, q;

   n = x.size();

   for (i = 0; i < n; i++) 
   {  sum_x  = sum_x  + x[i];
      sum_x2 = sum_x2 + x[i] * x[i];
      sum_y  = sum_y  + y[i];
      sum_xy = sum_xy + x[i] * y[i];
   }

   m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
   q = (sum_y - m * sum_x) / n;

   return {m,q};
}

// cost as SER
tuple<int,int,double,double,double> costSER(int low, int up, vector<double> y)
{  int i,n;
   double m,q;
   vector<int> x;
   vector<double> ypred,residuals;

   n = y.size();
   for(i=0;i<n;i++)
      x.push_back(low+i);

   tie(m, q) = linearRegression(x, y);
   for(i=0;i<n;i++)
   {
      ypred.push_back(m * x[i] + q);
      residuals.push_back(y[i] - ypred[i]);
   }

   // compute variance
   double sum = accumulate(begin(residuals), end(residuals), 0.0);
   double media = sum / residuals.size();
   double accum = 0.0;
   for_each(begin(residuals), end(residuals), [&](const double d) {
      accum += (d - media) * (d - media);
      });

   double variance = accum / (n - 1);
   double SERcost = sqrt(variance / n);
   return {low, up, m, q, SERcost};
}

// cost as chi2
tuple<int, int, double, double, double> costChi2(int low, int up, vector<double> y)
{
   int i, n;
   double m, q, r, sumres2=0, sumchi = 0;
   vector<int> x;
   vector<double> ypred, residuals;

   n = y.size();
   for (i = 0; i < n; i++)
      x.push_back(low + i);

   tie(m, q) = linearRegression(x, y);
   for (i = 0; i < n; i++)
   {
      ypred.push_back(m * x[i] + q);
      r = y[i] - ypred[i];
      residuals.push_back(r);
      sumres2 += r*r;
      sumchi  += r*r/ypred[i];
   }

   return { low, up, m, q, sumchi };
}

// cost as MSE
tuple<int, int, double, double, double> costMSE(int low, int up, vector<double> y)
{
   int i, n;
   double m, q, r, sumres2 = 0;
   vector<int> x;
   vector<double> ypred, residuals;

   n = y.size();
   for (i = 0; i < n; i++)
      x.push_back(low + i);

   tie(m, q) = linearRegression(x, y);
   for (i = 0; i < n; i++)
   {
      ypred.push_back(m * x[i] + q);
      r = y[i] - ypred[i];
      residuals.push_back(r);
      sumres2 += r * r;
   }
   double MSEcost = 0;
   MSEcost = sumres2 / n;
   return { low, up, m, q, MSEcost };
}

// cost as r squared
tuple<int, int, double, double, double> costR2(int low, int up, vector<double> y)
{
   int i, n;
   double m, q, r, rsd, sum, xbar, ybar, sumres2 = 0;
   double WN1 = 0, WN2 = 0, WN3 = 0, WN4 = 0, Sy = 0, Sx = 0;
   vector<int> x;
   vector<double> ypred, residuals;

   n = y.size();
   sum = 0;
   for (i = 0; i < n; i++)
   {  x.push_back(low + i);
      sum += low+i;
   }
   xbar = sum/n;

   tie(m, q) = linearRegression(x, y);
   sum = 0;
   for (i = 0; i < n; i++)
   {  ypred.push_back(m * x[i] + q);
      rsd = y[i] - ypred[i];
      residuals.push_back(rsd);
      sumres2 += rsd * rsd;
      sum += y[i];
   }
   ybar = sum/n;

   //Calculate r correlation
   for (i = 0; i < n; ++i) {
      WN1 += (x[i] - xbar) * (y[i] - ybar);
      WN2 += pow((x[i] - xbar), 2);
      WN3 += pow((y[i] - ybar), 2);
   }
   WN4 = WN2 * WN3;
   r = WN1 / (sqrt(WN4));

   double R2cost = r*r;
   return { low, up, m, q, R2cost };
}

// cost as variance
tuple<int, int, double, double, double> costVar(int low, int up, vector<double> y)
{
   int i, n;
   double m, q, r, sumres2 = 0, sumchi = 0;
   vector<int> x;
   vector<double> ypred, residuals;

   n = y.size();
   for (i = 0; i < n; i++)
      x.push_back(low + i);

   tie(m, q) = linearRegression(x, y);
   for (i = 0; i < n; i++)
   {
      ypred.push_back(m * x[i] + q);
      r = y[i] - ypred[i];
      residuals.push_back(r);
      sumres2 += r * r;
      sumchi += r * r / ypred[i];
   }

   // compute variance
   double sum = accumulate(begin(residuals), end(residuals), 0.0);
   double media = sum / residuals.size();
   double accum = 0.0;
   for_each(begin(residuals), end(residuals), [&](const double d) {
      accum += (d - media) * (d - media);
      });

   double variance = accum / (n - 1);
   return { low, up, m, q, variance };
}

// cost as RMSE
tuple<int, int, double, double, double> costRMSE(int low, int up, vector<double> y)
{
   int i, n;
   double m, q, r, sumres2 = 0, sumchi = 0;
   vector<int> x;
   vector<double> ypred, residuals;

   n = y.size();
   for (i = 0; i < n; i++)
      x.push_back(low + i);

   tie(m, q) = linearRegression(x, y);
   for (i = 0; i < n; i++)
   {  ypred.push_back(m * x[i] + q);
      r = y[i] - ypred[i];
      residuals.push_back(r);
      sumres2 += r * r;
      sumchi += r * r / ypred[i];
   }

   // compute variance
   double sum = accumulate(begin(residuals), end(residuals), 0.0);
   double media = sum / residuals.size();
   double accum = 0.0;
   for_each(begin(residuals), end(residuals), [&](const double d) {
      accum += (d - media) * (d - media);
      });

   double costRMSE = accum / sqrt(n - 1);
   return { low, up, m, q, costRMSE };
}

// cost as quasi RMSE
tuple<int, int, double, double, double> costQRMSE(int low, int up, vector<double> y)
{
   int i, n;
   double m, q, r, sumres2 = 0, sumchi = 0;
   vector<int> x;
   vector<double> ypred, residuals;

   n = y.size();
   for (i = 0; i < n; i++)
      x.push_back(low + i);

   tie(m, q) = linearRegression(x, y);
   for (i = 0; i < n; i++)
   {
      ypred.push_back(m * x[i] + q);
      r = y[i] - ypred[i];
      residuals.push_back(r);
      sumres2 += r * r;
   }
   double costQRMSE = sumres2 / sqrt(n);
   return { low, up, m, q, costQRMSE };
}

// cost as quasi RMSE multiplied by n
tuple<int, int, double, double, double> costQRMSEn(int low, int up, vector<double> y)
{  int i, n;
   double m, q, r, sumres2 = 0, sumchi = 0;
   vector<int> x;
   vector<double> ypred, residuals;

   n = y.size();
   for (i = 0; i < n; i++)
      x.push_back(low + i);

   tie(m, q) = linearRegression(x, y);
   for (i = 0; i < n; i++)
   {
      ypred.push_back(m * x[i] + q);
      r = y[i] - ypred[i];
      residuals.push_back(r);
      sumres2 += r * r;
   }
   double costQRMSEn = n*sumres2 / sqrt(n);
   return { low, up, m, q, costQRMSEn };
}

// Function to calculate residual sum of squares (RSS)
double calculateRSS(const std::vector<double>& y, const std::vector<double>& y_pred) 
{  double rss = 0.0;
   for (size_t i = 0; i < y.size(); ++i) {
      rss += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
   }
   return rss;
}

// cost as AIC
tuple<int, int, double, double, double> costAIC(int low, int up, vector<double> y) 
{  int num_params = 2;
   int i, n;
   double m, q, r, sumres2 = 0;
   vector<int> x;
   vector<double> ypred, residuals;

   n = y.size();
   for (i = 0; i < n; i++)
      x.push_back(low + i);

   tie(m, q) = linearRegression(x, y);  // corrisponde a y_pred
   for (i = 0; i < n; i++)
   {  ypred.push_back(m * x[i] + q);
      r = y[i] - ypred[i];
      residuals.push_back(r);
      sumres2 += r * r;
   }

   double aic = n * std::log(sumres2 / n) + 2 * num_params;

   // here it becomes AICc
   if(n<40)
      aic = aic + (2*num_params*num_params+2*num_params)/(n-num_params+1);
   return { low, up, m, q, aic};
}

// computes all feasible runs
vector<tuple<int, int, double, double, double>> computeRuns(int minlag, vector<double> y,int idcost)
{  int low,up,n,i,cont=0;
   vector<double> ytup;

   vector<tuple<int,int,double,double,double>> lstOLS;
   tuple<int, int, double, double, double> tup;

   n = y.size();
   for (low=0; low < (n - minlag + 1);low++)
      for (up= low + minlag - 1; up < n;up++)
      {  ytup.clear();
         for(i=low;i<up+1;i++)
            ytup.push_back(y[i]);

         switch (idcost)
         {
            case 0 : tup = costR2(low, up, ytup);     break;
            case 1 : tup = costMSE(low, up, ytup);    break;
            case 2 : tup = costChi2(low, up, ytup);   break; // low, up, m, q, cost, low and up included
            case 3 : tup = costSER(low, up, ytup);    break;
            case 4 : tup = costVar(low, up, ytup);    break;
            case 5 : tup = costRMSE(low, up, ytup);   break;
            case 6 : tup = costQRMSE(low, up, ytup);  break;
            case 7 : tup = costQRMSEn(low, up, ytup); break;
            case 8 : tup = costAIC(low, up, ytup);    break;
            default: cout << "------- ERROR IN computeRuns COST FUNCTION ----------";
         }
         lstOLS.push_back(tup);
         cont++;
         if (low % 250 == 0)
            cout << "Computing cost for low=" << low << " num cols=" << cont << endl;
      }
   return lstOLS;
}

// checks all rows are covered
bool checkLagrFeas(vector<int> t0, vector<int> t1, vector<int> xiter, vector<int>& numCover)
{  bool isFeasible = true;
   int i,j,totcovered=0;
   int n = t0.size();
   int m = numCover.size();

   for(i=0;i<m;i++) numCover[i] = 0;
   for (int jj = 0; jj < xiter.size(); jj++)
   {  j = xiter[jj];
      for(i=t0[j];i<=t1[j];i++)  // for each row covered by column j
      {  if(numCover[i] == 0) totcovered++;  // covered one more row
         numCover[i]++;
      }
   }
   isFeasible = (totcovered == m);

   return isFeasible;
}

// tiny test initializer
void fillList(vector<tuple<int, int, double, double, double>>& lstOLS)
{
   tuple<int, int, double, double, double> v = lstOLS[0];
   lstOLS.clear();
   for(int i=0;i<12;i++) lstOLS.push_back(v);
   get<0>(lstOLS[0]) = 0; get<1>(lstOLS[0]) = 4;  get<4>(lstOLS[0]) = 5;
   get<0>(lstOLS[1]) = 0; get<1>(lstOLS[1]) = 5;  get<4>(lstOLS[1]) = 7;
   get<0>(lstOLS[2]) = 1; get<1>(lstOLS[2]) = 4;  get<4>(lstOLS[2]) = 3;
   get<0>(lstOLS[3]) = 1; get<1>(lstOLS[3]) = 6;  get<4>(lstOLS[3]) = 2;
   get<0>(lstOLS[4]) = 2; get<1>(lstOLS[4]) = 5;  get<4>(lstOLS[4]) = 7;
   get<0>(lstOLS[5]) = 2; get<1>(lstOLS[5]) = 7;  get<4>(lstOLS[5]) = 5;
   get<0>(lstOLS[6]) = 3; get<1>(lstOLS[6]) = 8;  get<4>(lstOLS[6]) = 8;
   get<0>(lstOLS[7]) = 4; get<1>(lstOLS[7]) = 9;  get<4>(lstOLS[7]) = 6;
   get<0>(lstOLS[8]) = 4; get<1>(lstOLS[8]) = 10; get<4>(lstOLS[8]) = 3;
   get<0>(lstOLS[9]) = 5; get<1>(lstOLS[9]) = 10; get<4>(lstOLS[9]) = 5;
   get<0>(lstOLS[10]) = 6; get<1>(lstOLS[10]) = 9; get<4>(lstOLS[10]) = 2;
   get<0>(lstOLS[11]) = 6; get<1>(lstOLS[11]) = 10; get<4>(lstOLS[11]) = 4;
}

// lagrangian, print of iter subgradient, lambdas, ...
void printout(vector<int> xiter, vector<double> lambda, vector<int> subgr)
{  int i;
   cout << "xiter: ";
   for (i = 0; i < xiter.size(); i++) cout << xiter[i] << " ";
   cout << endl;

   cout << "subgr: ";
   for (i = 0; i < subgr.size(); i++) cout << subgr[i] << " ";
   cout << endl;

   cout << "lambda: ";
   for (i = 0; i < lambda.size(); i++) cout << lambda[i] << " ";
   cout << endl;
}

// first zub
double initZub(vector<double> obj, vector<int> t0, vector<int> t1)
{  int i,t,totCover=0;
   double zub=0;
   bool isXsol;
   int n = obj.size();
   int m = t1.size();
   int tmax = *max_element(begin(t1),end(t1));
   vector<int> numCover(tmax+1);
   i = 0;
   while (totCover < (tmax+1) && i<n)
   {  isXsol = false;
      for (t = t0[i]; t <= t1[i]; t++)
      {
         if (numCover[t] == 0)
         {  isXsol = true;
            numCover[t]++;
            totCover++;
         }
      }
      if(isXsol) zub+=obj[i];
      i++;
   }
   if (i == n)
   {  cout << "ERROR - could find no zub" << endl;
      zub = -1;
   }
   cout << "Initial zub " << zub << endl;
   return zub;
}

// fix xiter and get a feasible cost
pair<double,int> iterZub(vector<tuple<int, int, double, double, double>> lstOLS, int minlag, vector<int> xiter, vector<double> lambda, vector<double> c)
{  int i,j,k,m,n,newcol,t0,t1,tmax,idSegm,n_brk;
   double z;
   vector<int> xfeas;     // the corresponding feasible solution
   vector<int> initTimes; // tstart of each sement in xiter
   vector<int> idx;       // indices of xiter
   vector<int> innerSol;  // solution of DAG

   z = 0;
   n = colids.size();
   m = rowids.size();
   vector<bool> isCovered(m);
   for (int jj = 0; jj < xiter.size(); jj++)
   {  j = xiter[jj];
      z += c[j];
      initTimes.push_back(get<0>(lstOLS[xiter[jj]]));
      idx.push_back(jj);
      xfeas.push_back(j); // the seed of the feasible solution
      for (i = 0; i < colids[j].size(); i++)
         isCovered[colids[j][i]] = true;  // righe coperte in xiter
   }

   sortBasedOnAnother(initTimes,idx);

   // prune xiter of overlapping segments   
   i=0;
   while (i < ((int) idx.size())-1)
   {  t0 = get<1>(lstOLS[xiter[idx[i]]]);
      for(j=idx.size()-1;j>i;j--)
      {  t1 = get<0>(lstOLS[xiter[idx[j]]]);
         if(t1<t0+1)
            idx.erase(idx.begin() + j);
      }
      i++;
   }

   // redefine xiter containing only the pruned segments.
   xfeas.clear();
   for(i=0;i<idx.size();i++)
      xfeas.push_back(xiter[idx[i]]);
   xiter = xfeas;
   xfeas.clear();

   // compute coplete solution
   tmax = get<1>(lstOLS[lstOLS.size() - 1]);
   if (xiter.size() == 0)
   {  for (i = 0; i < lstOLS.size(); i++)
         if(get<0>(lstOLS[i]) == 0 && get<1>(lstOLS[i]) == tmax)
         {  z = get<4>(lstOLS[i]);
            xfeas.push_back( getSegmentId(0, tmax, lstOLS) ); // i.e., i
            break;
         }
   }
   else // fill all gaps between segments
   {  t0=0;
      for (i = 0; i < xiter.size(); i++)
      {  idSegm = xiter[i];
         t1 = get<0>(lstOLS[idSegm]); // init of the solution segment
         if(t1-t0 > minlag)
         {  innerSol = DAG_SSSP(t0, t1 - 1, minlag, lstOLS);
            for(j=0;j<innerSol.size();j++) z += get<4>(lstOLS[innerSol[j]]);
            xfeas.insert(xfeas.end(), innerSol.begin(), innerSol.end());
         }
         else if(i==0)  // extend first segment backwards
         {  j = getSegmentId(0,get<1>(lstOLS[xiter[i]]), lstOLS);
            idSegm = j;  // elongated segment
         }
         xfeas.push_back(idSegm);
         t0 = get<1>(lstOLS[idSegm]) + 1;
      }
      if(t0 < (tmax-minlag))
      {  innerSol = DAG_SSSP(t0, tmax, minlag, lstOLS);
         for (j = 0; j < innerSol.size(); j++) z += get<4>(lstOLS[innerSol[j]]);
         xfeas.insert(xfeas.end(), innerSol.begin(), innerSol.end());
      }

      // fill small gaps between segments. very rudely
      initTimes.clear(); idx.clear();
      for (i = 0; i < xfeas.size(); i++)
      {  initTimes.push_back(get<0>(lstOLS[xfeas[i]]));
         idx.push_back(i);
      }
      sortBasedOnAnother(initTimes, xfeas); // sort xfeas based on inittimes
      for (i = 0; i < xfeas.size()-1; i++)
      {  k = get<0>(lstOLS[xfeas[i+1]])-get<1>(lstOLS[xfeas[i]]);
         if(k > 1)
         {  t1 = get<1>(lstOLS[xfeas[i]])+k-1;
            j = getSegmentId(get<0>(lstOLS[xfeas[i]]),t1,lstOLS);
            xfeas[i] = j;  // elongated segment
         }
      }
      // the last gap, if any
      if( get<1>(lstOLS[xfeas[i]])<tmax )
      {  k = tmax - get<1>(lstOLS[xfeas[i]]);
         j = getSegmentId(get<0>(lstOLS[xfeas[i]]), tmax, lstOLS);
         xfeas[i] = j;  // elongated segment
      }
      z = 0;
      for (i = 0; i < xfeas.size(); i++)
         z += get<4>(lstOLS[xfeas[i]]);
   }
   //cout << "zubiter " << z << endl;

   bool isFeas = checkFeas(xfeas,lstOLS,z);
   if(!isFeas) 
      throw std::invalid_argument("[iterZub] infeasible solution.");
   n_brk = xfeas.size()-1; // breakpoints are num of segments minus 1
   return make_pair(z,n_brk);
}

void printTableau(int n, int m,vector<double> obj,vector<int> t0, vector<int> t1)
{
   int i,j;
   int aij;
   ofstream dsFile("tableau.csv");
   for(j=0;j<n;j++)
      dsFile << obj[j] << " ";
   dsFile << endl;
   for(i=0;i<m;i++)
   {  for (j = 0; j < n; j++)
      {  aij = 0;
         if (i >= t0[j] && i <= t1[j]) aij = 1;
         dsFile << aij << " ";
      }
      dsFile << endl;
   }
   dsFile.close();
}

// Function to sort both vectors based on initial values of vector v1. If v2 idx it has the sorted indices
void sortBasedOnAnother(std::vector<int>& v1, std::vector<int>& v2) 
{  // Check if the vectors are of the same size
   if (v1.size() != v2.size()) 
      throw std::invalid_argument("[sortBasedOnAnother] Vectors must be of the same length.");

   // Create a vector of pairs
   vector<std::pair<int, int>> pairedVector;
   for (size_t i = 0; i < v1.size(); ++i) 
      pairedVector.emplace_back(v1[i], v2[i]);

   // Sort the vector of pairs based on the first element of the pair
   sort(pairedVector.begin(), pairedVector.end(),
      [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
         return a.first < b.first;
      });

   // Separate the pairs back into the original vectors
   for (size_t i = 0; i < pairedVector.size(); ++i) {
      v1[i] = pairedVector[i].first;
      v2[i] = pairedVector[i].second;
   }
}

// y has sensor values
double lagrangian(int maxruns, vector<tuple<int, int, double, double, double>> lstOLS, int minlag, double alpha)
{  int i,j, iter, sumSubgr2, n_brk=-1;
   vector<double> c;       // lagrangian, penalized costs at each iteration
   vector<double> obj;     // the original costs
   vector<double> lambda;
   vector<int>    idx;
   vector<int>    xiter;
   vector<int>    t0,t1;   // initial and final time of segments
   vector<int>    numCover;// num of cols covering row i in the sol
   vector<int>    subgr;   // iteration subgradients
   vector<int>    x;       // 0/1 repr of the solution
   double  cost,zlb=0,zlbIter,zubIter,sumLambda,step,zub=0,eps=0.000001,tzub=-1;
   bool    isFeasible;
   clock_t tstart, tLagr;

   //fillList(lstOLS);
   int n = lstOLS.size();

   // create the columns.
   for(j=0;j<n;j++) 
   {  obj.push_back(get<4>(lstOLS[j]));
      t0.push_back(get<0>(lstOLS[j]));
      t1.push_back(get<1>(lstOLS[j]));
      idx.push_back(j);
      x.push_back(0);
   }

   int tmax = *max_element(begin(t1), end(t1));
   int m = tmax + 1;  // useful for loops
   zub = initZub(obj, t0, t1);

   for(i=0;i<m;i++)
   {  lambda.push_back(0);
      numCover.push_back(0);
      subgr.push_back(0);
   }

   iter = 0;
   tstart= clock();
   while(iter < maxIter)
   {  tLagr = clock();
      double ttot = (tLagr - tstart) / CLOCKS_PER_SEC;
      if(ttot > maxTime)
         break;
      c.clear();
      for(j=0;j<n;j++) x[j]=0;
      for(i=0;i<xiter.size();i++) x[xiter[i]]=1; // 0/1 solution
      for (j = 0; j < n; j++)
      {  cost = obj[j];
         for (i = 0; i < m; i++)
         {  if(i>=t0[j] && i<=t1[j])
               cost -= lambda[i];
         }
         c.push_back(cost);
      }
      zlbIter = 0;
      // sorting indices of obj
      //iota(idx.begin(), idx.end(), 0); // Initializing idx, increasing values
      sort(idx.begin(), idx.end(), [&](int i, int j) {return c[i] < c[j]; });

      i = sumLambda = 0;
      xiter.clear();
      while (i<n && c[idx[i]] < 0 && i < maxruns)
      {  //cout << "  var " << i << " entering " << idx[i] << " value " << c[idx[i]] << endl;
         zlbIter += c[idx[i]];
         xiter.push_back(idx[i]);
         i++;
      }

      for(j=0;j<m;j++) sumLambda += lambda[j];
      zlbIter += sumLambda;
      if(zlbIter > zlb) zlb = zlbIter;
      std::pair<double,int> result = iterZub(lstOLS,minlag,xiter,lambda,obj);
      zubIter = result.first;
      int n_brkIter = result.second;
      if(zubIter < zub) 
      {  cout << "--- new zub: " << zubIter << endl;
         zub = zubIter;
         n_brk = n_brkIter;
         tzub = ttot;
      }
      if(iter%10 == 0 || iter==maxIter-1)
         cout << "iteration " << iter << " zlb " << zlb << " zlbIter: " << zlbIter << 
                 " zubIter: " << zubIter << " zub " << zub << " n_brk " << n_brk << 
                 " tzub " << tzub << endl;

      isFeasible = checkLagrFeas(t0,t1,xiter,numCover);
      if (isFeasible)
      {  zubIter = 0;
         for(i=0;i<xiter.size();i++) zubIter += obj[xiter[i]];
         if(zubIter<zub) 
         {  cout << "--- new zub: " << zubIter << endl;
            zub=zubIter;
         }
         if(abs(zubIter-zlb) < eps)
         {  cout << "OPTIMUM!!! " << zubIter << " eps " << eps << endl;
            printout(xiter, lambda, subgr);
            return zub;
         }
      }
      // penalty update
      sumSubgr2 = 0;
      for (i = 0; i < m; i++)
      {  subgr[i]  = 1-numCover[i];
         sumSubgr2 += subgr[i]*subgr[i];
      }

      double zz = zub;
      if(zlbIter > 0) zz = min(1.5*zlbIter,zub);

      step = alpha*(zz-zlbIter)/sumSubgr2;
      for(i=0;i<m;i++)
         lambda[i] = max(0.,lambda[i]+step*subgr[i]);
      //cout << " Sumsubgr2 " << sumSubgr2 << " step " << step << endl;
      //printout(xiter, lambda, subgr);
      iter++;
   }
   return zub;
}

void postProcess(vector<tuple<int, int, double, double, double>>& lstOLS, vector<double>& x, int minlag)
{  int i,j,i1,i2,n,x10,x11,x20,x21,imin,imax,segm1,segm2;
   double y11,y20,m1,m2,q1,q2,intersecX,intersecY,cost;
   tuple<int, int, double, double, double> tup;

   n = lstOLS.size();
   vector<int> idLine;  // solution, segments id
   for (i = 0; i < n; i++)
      if(x[i] > 0.1)
         idLine.push_back(i); // segment is in the solution

   for (i = 0; i < idLine.size() - 1; i++)
   {  x10 = get<0>(lstOLS[idLine[i]]);
      x11 = get<1>(lstOLS[idLine[i]]);
      x20 = get<0>(lstOLS[idLine[i + 1]]);
      x21 = get<1>(lstOLS[idLine[i + 1]]);
      i1 = idLine[i];
      i2 = idLine[i+1];
      if(x11>x20)
      {
         if(x11>=x21)
         {  // i contiene completamente i+1. La si copia su i+1
            cout << "-- overlapping segments " << idLine[i] << " and " << idLine[i+1] << " keeping " << idLine[i] << endl;
            idLine[i+1] = idLine[i];
            idLine[i]   = -1;
            continue;
         }
         if ( (x21 - x10 < 2*minlag+1) )
         {  x[11] = 0;
            x[i2] = 0;
            j = getSegmentId(x10, x21, lstOLS);
            // found the segment long as the two united
            x[j] = 1;
            idLine[i] = -1; // idLine[i] to be deleted, but inifluential (solution in x)
            idLine[i+1] = j; 
            cout << "-- overlapping segments " << i1 << " and " << i2 << " too short to cut, union into " << j << endl;
         }
         else // need to redefine endpoints
         {  imin = x10 + minlag; // earliest cutpoint (init 2nd segment)
            imax = x21 - minlag + 1; // latest cutpoint
            if(imin<=imax) 
            {
               double mincost = DBL_MAX;
               int    minj = -1;
               for (j = imin; j < imax; j++)  // try all feasible cutss and keep the best one
               {  segm1 = getSegmentId(x10, j-1, lstOLS);
                  segm2 = getSegmentId(j, x21, lstOLS);
                  cost = get<4>(lstOLS[segm1]) + get<4>(lstOLS[segm2]);
                  if (cost < mincost)
                  {  mincost = cost;
                     minj    = j;
                  }
               }
               segm1 = getSegmentId(x10, minj-1, lstOLS);
               segm2 = getSegmentId(minj, x21, lstOLS);
               x[i1] = 0;
               x[i2] = 0;
               x[segm1] = 1;
               x[segm2] = 1;
               idLine[i]   = segm1;
               idLine[i+1] = segm2;
               cout << "-- overlapping segments " << i1 << " and " << i2 << " substituted by " << segm1 << " and " << segm2 << endl;
            }
            else
            {  // imin<<=>imax
               cout << "unmanaged case"<<endl;
            }
         }
      }
   }
   // removing the -1
   idLine.erase(std::remove(idLine.begin(), idLine.end(), -1), idLine.end());
   for (i = 0; i < n; i++)
      x[i] = 0;
   for (i=0;i<idLine.size();i++)
      x[idLine[i]] = 1.0; // segment is in the solution
}

// gets the segment id given its endpoint coords. Sequential, can be improved
int getSegmentId(int x0, int x1, vector<tuple<int, int, double, double, double>> lstOLS)
{  int j,n;

   n = lstOLS.size();
   for (j = 0; j < n; j++)
      if (get<0>(lstOLS[j]) == x0 && get<1>(lstOLS[j]) == x1)
         break;

   return j;
}

// copied from somewhere on the internet
int get_line_intersection(double p0_x, double p0_y, double p1_x, double p1_y, double p2_x, 
                          double p2_y, double p3_x, double p3_y, double* i_x, double* i_y)
{  double s02_x, s02_y, s10_x, s10_y, s32_x, s32_y, s_numer, t_numer, denom, t;
   s10_x = p1_x - p0_x;
   s10_y = p1_y - p0_y;
   s02_x = p0_x - p2_x;
   s02_y = p0_y - p2_y;

   s_numer = s10_x * s02_y - s10_y * s02_x;
   if (s_numer < 0)
      return 0; // No collision

   s32_x = p3_x - p2_x;
   s32_y = p3_y - p2_y;
   t_numer = s32_x * s02_y - s32_y * s02_x;
   if (t_numer < 0)
      return 0; // No collision

   denom = s10_x * s32_y - s32_x * s10_y;
   if (s_numer > denom || t_numer > denom)
      return 0; // No collision

   // Collision detected
   t = t_numer / denom;
   if (i_x != NULL)
      *i_x = p0_x + (t * s10_x);
   if (i_y != NULL)
      *i_y = p0_y + (t * s10_y);

   return 1;
}

// write csv file with candidate segments
void writeListOLS(vector<tuple<int, int, double, double, double>> lstOLS, string dsName)
{  int j;
   ofstream dsFile(baseDir + dsName + "_runs.csv");
   dsFile << "id,low,hi,m,q,cost"<< endl;
   for(j=0;j<lstOLS.size();j++)
      dsFile << j << "," << get<0>(lstOLS[j]) << ","
      << get<1>(lstOLS[j]) << ","
      << get<2>(lstOLS[j]) << ","
      << get<3>(lstOLS[j]) << ","
      << get<4>(lstOLS[j]) << endl;
   dsFile.close();
}

// rows in each column and columns covering each row
void compressTableau(vector<tuple<int, int, double, double, double>> lstOLS)
{  int i,j;
   int n = lstOLS.size();             // num cols
   vector<double> c;                  // costi
   for (j = 0; j < n; j++) 
   {  colids.push_back({});
      c.push_back(get<4>(lstOLS[j]));
   }
   int m = get<1>(lstOLS[n-1]) + 1;   // num rows
   for (i = 0; i < m; i++) rowids.push_back({});

   vector<int> idx(n);
   iota(idx.begin(), idx.end(), 0);   // Initializing idx, increasing values
   sort(idx.begin(), idx.end(), [&](int i, int j) {return c[i] < c[j]; }); // idx by incewasing costs


   for (int jj = 0; jj < n; jj++)
   {  j = idx[jj];                    // insert ordered by decreasing costs
      for (i = get<0>(lstOLS[j]); i < get<1>(lstOLS[j]) + 1; i++)
      {  colids[j].push_back(i);
         rowids[i].push_back(j);
      }
   }
}

// datafile etc.
int readConfig()
{
   int i, j, idcost;
   string line;

   //cout << "Running from " << exePath() << endl;

   ifstream fconf("config.json");
   stringstream buffer;
   buffer << fconf.rdbuf();
   line = buffer.str();
   json::Value JSV = json::Deserialize(line);

   baseDir = JSV["basedir"].ToString();
   dsName  = JSV["dsName"].ToString();
   maxIter = JSV["maxIter"].ToInt();
   maxTime = JSV["maxTime"].ToInt();
   idcost  = JSV["idcost"].ToInt();
   cout << baseDir << endl;
   cout << dsName << endl;
   return idcost;
}

int main(int argc, char** argv)
{  bool     fLagrangian = false;
   int      solstat, n_brk=-1;
   double   objval=-1, tCpuOpt, cost = 0;

   vector<double> x;
   vector<double> pi;
   vector<double> slack;
   vector<double> dj;
   vector<char>   ctype;

   CPXENVptr     env = NULL;
   CPXLPptr      lp = NULL;
   int           status = 0;
   int           i, j, n, idcost, cont;
   int           cur_numrows=-1, cur_numcols=-1;
   clock_t       tstart, truns, tMIP;

   std::cout << std::fixed; // prevent scientific notation output
   idcost = readConfig();   // cost function:  R2, MSE, Chi2, SER, var, RMSE, QRMSE, QRMSEn, AIC

   double alpha = 0.05;
   string dataFile = baseDir + dsName + ".csv";
   string segmentFileName = baseDir + dsName + "_runs.csv";

   string costFunc;
   ofstream dsFile(baseDir + dsName + "_segments.csv");
   vector<int> ids;
   vector<double> y;
   vector<tuple<int, int, double, double, double>> lstOLS;

   n = readData(dataFile, ids, y);
   cout << "read " << n << " values" << endl;
   int minlag = max(5, n / 20);

   tstart = clock();
   lstOLS = computeRuns(minlag, y, idcost);
   switch (idcost)
   {
      case 0: costFunc = "costR2"; break;
      case 1: costFunc = "costMSE"; break;
      case 2: costFunc = "costChi2"; break;
      case 3: costFunc = "costSER"; break;
      case 4: costFunc = "costVar"; break;
      case 5: costFunc = "costRMSE"; break;
      case 6: costFunc = "costQRMSE"; break;
      case 7: costFunc = "costQRMSEn"; break;
      case 8: costFunc = "costAIC"; break;
      default: cout << "ERROR 1" << endl;
   }
   n = lstOLS.size();
   truns = clock();
   double tCpuRuns = (truns - tstart) / CLOCKS_PER_SEC;
   cout << "CPU time for runs: " << tCpuRuns << endl;

   writeListOLS(lstOLS, dsName); // write csv file with candidate segments
   compressTableau(lstOLS);      // calcola tableau per righe e per colonne, compresse (lista indici)
   truns = clock();

   if (fLagrangian)
   {  cost = lagrangian(INT_MAX, lstOLS, minlag, alpha);
      tMIP = clock();
      tCpuOpt = (tMIP - truns) / CLOCKS_PER_SEC;
      cout << "CPU time for Lagr: " << tCpuOpt << endl;
      goto TERMINATE;
   }

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
   {
      cout << "Failed to create LP." << endl;
      goto TERMINATE;
   }

   // Now populate the problem with the data.
   status = populatebyrow(env, lp, y, lstOLS);
   if (status)
   {
      cout << "Failed to populate problem." << endl;
      goto TERMINATE;
   }

   // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> LP <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
   tCpuOpt = (tMIP - truns) / CLOCKS_PER_SEC;
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

   postProcess(lstOLS, x, minlag);

   cont = 0;
   dsFile << "id,low,hi,m,q," << costFunc << endl;
   for (j = 0; j < cur_numcols; j++)
      if (x[j] > 0.01)
      {
         cout << cont << ") column " << j << " value=" << x[j];
         cout << " segm: " << get<0>(lstOLS[j]) << ","
            << get<1>(lstOLS[j]) << ","
            << get<2>(lstOLS[j]) << ","
            << get<3>(lstOLS[j]) << ","
            << get<4>(lstOLS[j]) << endl;
         dsFile << j << "," << get<0>(lstOLS[j]) << ","
            << get<1>(lstOLS[j]) << ","
            << get<2>(lstOLS[j]) << ","
            << get<3>(lstOLS[j]) << ","
            << get<4>(lstOLS[j]) << endl;
         cont++;
         cost += get<4>(lstOLS[j]);
      }
   dsFile.close();
   n_brk = cont - 1;
   cout << "Final number of segments: " << cont << " cost " << cost << endl;

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
   {
      status = CPXcloseCPLEX(&env);
      if (status)
      {
         char  errmsg[CPXMESSAGEBUFSIZE];
         cout << "Could not close CPLEX environment." << endl;
         CPXgeterrorstring(env, status, errmsg);
         cout << errmsg << endl;
      }
   }

   cout << (fLagrangian ? "Lagrangian " : "MIP ") << dsName <<
      " n_runs " << n << " t_runs " << tCpuRuns <<
      " SCP objval " << objval << " final cost " << std::setprecision(6) << cost << " n_brk " << n_brk << " t_opt " << tCpuOpt <<
      " n_rows " << cur_numrows << " n_cols " << cur_numcols << endl;
   return (status);
}  /* END main */
