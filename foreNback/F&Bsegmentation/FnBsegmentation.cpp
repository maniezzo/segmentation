#include "FnBsegmentation.h"
#include "Stage.h"

// n number of points, maxt maximum time (=n-1)
void FnBsegmentation::run_FnB()
{  int i;
   bool isImprovedF = true, isImprovedB = true;

   // 2. Assign the pointer based on idcost
   switch (idcost) 
   {
      //case 0 : pntCost = &FnBsegmentation::costR2;     break;
      //case 1 : pntCost = &FnBsegmentation::costMSE;    break;
      //case 2 : pntCost = &FnBsegmentation::costChi2;   break; 
      //case 3 : pntCost = &FnBsegmentation::costSER;    break;
      //case 4 : pntCost = &FnBsegmentation::costVar;    break;
      //case 5 : pntCost = &FnBsegmentation::costRMSE;   break;
      case 6 : pntCost = &FnBsegmentation::costQRMSE; costName = "QRMSE"; break;
      //case 7 : pntCost = &FnBsegmentation::costQRMSEn; break;
      case 8 : pntCost = &FnBsegmentation::costAIC; costName = "AIC"; break;
      //case 9 : pntCost = &FnBsegmentation::costBIC;    break;
      default: cout << "------- ERROR IN ASSIGNING COST FUNCTION ----------";
   }

   Stage N;
   N.mainNode();

   cout << "\n---------------------------------------------------- DAG" << endl;
   // optimal solution with no constraint on the number of arcs (but with minLength)
   DAG_SSSP();

   cout << "\n---------------------------------------------------- BF" << endl;
   // optimal solution with one constraint on the number of arcs and with minLength
   run_BF();

   cout << "\n---------------------------------------------------- FnB" << endl;
   Fstage.resize(n);
   Fexpanded.resize(n);
   Fstage[0].insert(0, 0, {0});   // num,cost, [changepoints]

   Bstage.resize(n);
   Bexpanded.resize(n);
   Bstage[n-1].insert(0, 0, {n-1}); // num,cost, [changepoints]

   tstart = clock();
   do
   {  isImprovedF = forward();
      isImprovedB = backward();
      tend = clock();
      ttot = (tend - tstart) / CLOCKS_PER_SEC;
   } while((isImprovedF || isImprovedB) && ttot < maxcpu);
   cout << "FnB: t.cpu " << ttot << endl;
   reconstructFnBsolution();
}

// forward pass
bool FnBsegmentation::forward()
{  int i,j,t,t2,k,nSegm,tend;
   Stage seed;
   double z,lb;  // lower bound to completion
   bool isImproved = false, hasMatch;
   vector<int> lstPoints;
   tend = min(n,maxLength);

   for(i=0;i<n;i++)
   {  if(i>0 && i<minLength) continue;  // non ho segmenti più corti di minLength

      for(k=0;k<delta;k++) // beam width
      {
         z         = get<0>( Fstage[i].queryMinCost(maxNumEdges) ); // cost up to stage i
         nSegm     = get<1>( Fstage[i].queryMinCost(maxNumEdges) ); // num of segments up to stage i
         lstPoints = get<2>( Fstage[i].queryMinCost(maxNumEdges) ); // changepoints up to stage i

         if(Fstage[i].isEmpty())
            continue;
         // remove from unexpanded and add to expanded
         Fexpanded[i].insert(nSegm, z, lstPoints);
         auto res = Fstage[i].popMinCost(maxNumEdges);

         hasMatch = match(true, i, nSegm, z);
         if(hasMatch) continue;

         if (i<(n-minLength))
         {  lb = DBL_MAX;
            for(t2=i+1;t2<n;t2++)
               if(!Bstage[t2].isEmpty())
                  lb = min( lb,get<0>( Bstage[i+1].queryMinCost(maxNumEdges))); // minimum of unexpanded
            if(lb==DBL_MAX) lb=0;
         }
         else
            lb = 0;

         if(z + lb >= zub)
         {  nFathomed++;
            continue;
         }

         if(nSegm < maxNumEdges)
            for(t=i+minLength;t<tend;t++)
            {  bool fGen  = generateFoffspring(i, t, nSegm, z, lstPoints);
               isImproved = isImproved || fGen;
            }
      }
   }
   return isImproved;
}

// backward pass
bool FnBsegmentation::backward()
{  int i,j,t,t1,k,nSegm, tstart;
   Stage seed;
   double z,lb;  // lower bound to completion
   bool isImproved = false, hasMatch;
   vector<int> lstPoints;

   int maxt = n-1;
   for(i=maxt;i>=0;i--)
   {  if (i<maxt && (maxt-i)<minLength) continue;  // non ho segmenti più corti di minLength

      for(k=0;k<delta;k++) // beam width
      {  
         z         = get<0>( Bstage[i].queryMinCost(maxNumEdges) ); // cost up to stage i
         nSegm     = get<1>( Bstage[i].queryMinCost(maxNumEdges) ); // num of segments up to stage i
         lstPoints = get<2>( Bstage[i].queryMinCost(maxNumEdges) ); // changepoints up to stage i

         if(Bstage[i].isEmpty())
            continue;
         // remove from unexpanded and add to expanded
         Bexpanded[i].insert(nSegm, z, lstPoints);
         auto res = Bstage[i].popMinCost(maxNumEdges);

         hasMatch = match(false, i, nSegm, z);
         if(hasMatch) continue;

         if (i>minLength)
         {  lb = DBL_MAX;
            for(t1=i-1;t1>=0;t1--)
               if(!Fstage[t1].isEmpty())
                  lb = min( lb,get<0>( Fstage[t1].queryMinCost(maxNumEdges) )); // minimum of unexpanded
            if(lb==DBL_MAX) lb=0;
         }
         else
            lb = 0;

         if(z + lb >= zub)
         {  nFathomed++;
            continue;
         }

         if(nSegm < maxNumEdges)
         {  tstart = max(0,i-minLength-maxLength);
            for (t=i-minLength;t>=0;t--)
            {  bool fGen = generateBoffspring(i, t, nSegm, z, lstPoints); // backward, t<i
               isImproved = isImproved || fGen;
            }
         }
      }
   }
   return isImproved;
}

// forward offspring generation, cost computation
bool FnBsegmentation::generateFoffspring(int t1, int t2, int nSegm, double cost, vector<int> lstPoints)
{  double c,m,q,z;
   bool isImproved = false;

   tuple<int,int,double,double,double> res = (this->*pntCost)(t1, t2); //costQRMSE(t1, t2);
   z = cost+get<4>(res);
   lstPoints.push_back(t2);

   if (t2==(n-1) && z<zub)
   {  zub = z;
      changepoints = lstPoints;
      if(isVerbose)
      {  ttot = (clock()-tstart)/CLOCKS_PER_SEC;
         cout << "F) New zub: "<< zub << " t.cpu " << ttot << endl;
      }
      isImproved = true;
   }

   if(z < zub)
      Fstage[t2].insert(nSegm+1, z, lstPoints);
   else
      nFathomed++;
   //cout << "t1=" << t1 << " t2=" << t2 << " nSegm="<< nSegm+1 << " cost " << cost+z << endl;

   return isImproved;
}

// backward offspring generation, cost computation
bool FnBsegmentation::generateBoffspring(int t2, int t1, int nSegm, double cost, vector<int> lstPoints)
{  int i;
   double c,m,q,z;
   bool isImproved = false;

   tuple<int,int,double,double,double> res = (this->*pntCost)(t1, t2); //costQRMSE(t1, t2);
   z = cost+get<4>(res);
   lstPoints.insert(lstPoints.begin(), t1);

   if (t1==0 && z<zub)
   {  zub = z;
      changepoints = lstPoints;
      if(isVerbose)
      {  ttot = (clock()-tstart)/CLOCKS_PER_SEC;
         cout<<"B) New zub: "<<zub<<" t.cpu "<<ttot<<endl;
      }
      isImproved = true;
   }

   if(z < zub)
      Bstage[t1].insert(nSegm+1, z,lstPoints);
   else
      nFathomed++;
   //cout << "t1=" << t1 << " t2=" << t2 << " nSegm="<< nSegm+1 << " cost " << cost+z << endl;

   return isImproved;
}

// matches the current partial solution against one from the opposite direction
bool FnBsegmentation::match(bool isForward, int i, int numSegm, double z)
{  double lb; // cost of completion
   bool hasMatch = false;

   int maxt = n-1;
   if (isForward)
      if(i<maxt)
      {  lb = get<0>( Bexpanded[i+1].queryMinCost(maxNumEdges-numSegm) );
         if(lb==0) goto l0; // no feasible expansion
         hasMatch = true;
         numMatch++;
         if(z + lb < zub)
         {  // new incumbent
            zub = z + lb;
            changepoints = get<2>(Fexpanded[i].queryMinCost(numSegm));
            vector<int> vb = get<2>(Bexpanded[i+1].queryMinCost(maxNumEdges-numSegm));
            changepoints.insert(changepoints.end(), vb.begin()+1, vb.end()); // Append vb
            std::sort(changepoints.begin(), changepoints.end()); // Sort the merged vector
            if(isVerbose)
               cout << "F) Match: new zub: "<< zub << endl;
         }
      }
   else
      if(i>0)
      {  lb = get<0>( Fexpanded[i-1].queryMinCost(maxNumEdges-numSegm) );
         if(lb==0) goto l0; // no feasible expansion
         hasMatch = true;
         numMatch++;
         if(z + lb < zub)
         {  // new incumbent
            zub = z + lb;
            changepoints = get<2>(Bexpanded[i].queryMinCost(numSegm));
            vector<int> vb = get<2>(Fexpanded[i-1].queryMinCost(maxNumEdges-numSegm));
            changepoints.insert(changepoints.end(), vb.begin()+1, vb.end()); // Append vb
            std::sort(changepoints.begin(), changepoints.end()); // Sort the merged vector
            if(isVerbose)
               cout << "B) Match: new zub: "<< zub << endl;
         }
      }
l0:return hasMatch;
}

// ricostruisce la soluzione FnB
void FnBsegmentation::reconstructFnBsolution()
{  int i;
   double sum = 0;
   vector<tuple<int, int, double, double, double>> sol;
   ttot = (clock() - tstart) / CLOCKS_PER_SEC;

   cout << "Changepoints: ";
   int t1 = 0;
   for (i=1;i<changepoints.size();i++)
   {  int t2 = changepoints[i];
      tuple<int, int, double, double, double> tup = (this->*pntCost)(t1, t2); //costQRMSE(t1,t2);
      sum += get<4>(tup);
      sol.push_back(tup);
      cout << t2 << " ";
      t1 = t2;
   }
   cout << endl;
   cout << "Costo complessivo " << sum << " t.cpu " << ttot << " num.matches " << numMatch << " n.fathomed " << nFathomed << endl;
   writeSolCsv(sol,"test_sol.csv");
}

// single source on a DAG. n number of points, maxt maximum time (=n-1)
void FnBsegmentation::DAG_SSSP()
{  int i,j,t,currInit,maxt,maxstart,tend;
   double c;
   tuple<int, int, double, double, double> tup;
   vector<tuple<int, int, double, double, double>> lstOLS; // t1,t2,m,q,cost of the segment
   vector<tuple<int, int, double, double, double>> sol;

   maxt     = n-1;
   maxstart = maxt - minLength;
   lstOLS.resize(n);
   tstart   = clock();

   lstOLS[0] = tuple<int, int, double, double, double>(0,0,0,0,0);
   vector<double> mincost(n,DBL_MAX); // min cost for covering up to time t
   mincost[0] = 0;

   for (t=0;t<=maxstart;t++)
   {  tend = min(maxt,t+maxLength);
      for(j=t+minLength;j<=tend;j++)
      {  if(t>0 && t<minLength) continue;
         tup = (this->*pntCost)(t,j); //costQRMSE(t,j);  // cosi' segmenti attaccati, se staccati da t a j-1
         c   = get<4>(tup);
         if (mincost[j]==DBL_MAX || mincost[j]>(mincost[t]+c))
         {  mincost[j] = mincost[t] + c;
            lstOLS[j]  = tup;
         }
      }
   }
   tend = clock();
   ttot = (tend - tstart) / CLOCKS_PER_SEC;

   sol = reconstructDAGsolution(lstOLS,maxt);
   writeSolCsv(sol,"test_sol.csv");
   cout << "DAG " << dsName << " cost: " << std::setprecision(5) << mincost[maxt] << " n_brk " << sol.size()-1 << " t.cpu " << ttot << endl;
}

// ricostruisce la soluzione DAG
vector<tuple<int, int, double, double, double>> FnBsegmentation::reconstructDAGsolution(vector<tuple<int, int, double, double, double>> lstOLS, int maxt)
{  int i,j,t;
   double sum = 0;
   vector<tuple<int, int, double, double, double>> sol;

   t = maxt;
   while(t>0)
   {  i    = get<0>(lstOLS[t]);  // inizio segmento ottimo che arriva in t
      sum += get<4>(lstOLS[t]);  // costo integrale
      sol.push_back(lstOLS[t]);
      cout << "Segm " << t << ") t1=" << get<0>(lstOLS[t]) << " t2= " << get<1>(lstOLS[t]) << 
         " m= " << get<2>(lstOLS[t]) << " q= " << get<3>(lstOLS[t]) <<" costo " << get<4>(lstOLS[t]) <<endl;
      t = i;
   }
   cout << "Costo complessivo " << sum << endl;
   return sol;
}

// imposta poi lancia bellman ford. n number of points, maxt maximum time (=n-1)
int FnBsegmentation::run_BF()
{  int cont,tend;
   int numEdges = 0;
   int numv     = n; // partono da 0
   vector<Edge> edges;
   vector<tuple<int, int, double, double, double>> lstOLS;
   vector<tuple<int, int, double, double, double>> sol;

   cout << "For each edge:" << endl;
   cont=0;
   for (int t1 = 0; t1 < n-minLength; ++t1) 
   {  if(t1>0 && t1<minLength) continue;
      tend = min(n,t1+maxLength);
      for (int t2 = t1+minLength; t2 < tend; ++t2) 
      {  if(t2<n-1 && t2 > n-minLength) continue;
         Edge edge;
         edge.end1 = t1;
         edge.end2 = t2;
         lstOLS.push_back((this->*pntCost)(t1, t2)); //costQRMSE(t1,t2));
         edge.cost = get<4>(lstOLS[cont]);
         edge.segm = cont++;
         edges.push_back(edge);
      }
   }

   sol = bellmanFord(edges, numv, maxNumEdges);
   writeSolCsv(sol, "test_sol.csv");
   return 0;
}

// Bellman-Ford algorithm with bounded number of edges
vector<tuple<int, int, double, double, double>> FnBsegmentation::bellmanFord(vector<Edge>& edges, int numv, int maxNumEdges) 
{  int i,j,u,v,t;
   double w,du;
   vector<double> cost0(numv, DBL_MAX);
   vector<double> cost1(numv, DBL_MAX); // cost matrix after iteration update
   vector<vector<int>> paths0(numv); // final paths
   vector<vector<int>> paths1(numv); // final paths after iteration update
   vector<int> prev(numv, -1);
   vector<int> minsegm(numv, -1);
   vector<tuple<int, int, double, double, double>> sol;
   tuple<int, int, double, double, double> tup;

   tstart = clock();
   for(i=0;i<numv;i++)
      paths0[i] = vector<int>();
   cost0[0] = 0;

   
   for (i = 0; i < min(maxNumEdges,numv-1); ++i) // stages. last one would be for checking negative cycles
   {  for (j=0;j<edges.size();j++)
      {  u = edges[j].end1;
         v = edges[j].end2;
         w = edges[j].cost;

         du = 0;
         if(u>0) du = cost0[u]; // il nuovo segm parte subito dopo la fine del precedente
         //if(u>0) du = cost0[u-1]; // il nuovo segm parte 1 dopo la fine del precedente
         if (cost0[u] != DBL_MAX && (du + w) < cost1[v])
         {  cost1[v] = du + w;
            prev[v] = u;
            minsegm[v]  = edges[j].segm;
            paths1[v] = (u==0 ? paths0[u] : paths0[u]);
            paths1[v].push_back(j); // l'arco percorso per arrivare
         }
      }
      cost0 = cost1; // cost1 will be the cost0 at next iteration
      paths0 = paths1;
   }
   tend = clock();
   ttot = (tend - tstart) / CLOCKS_PER_SEC;

   // Check for negative weight cycles: useless in DAG

   // Print shortest path distances
   double totc = 0;
   int nedges = 0;
   cout << "BF, Shortest path cost " << cost1[numv-1] << " with " << maxNumEdges << " arcs" << endl;
   for(i=0;i<paths1[numv-1].size();i++)
   {  totc += edges[paths1[numv - 1][i]].cost;
      cout << i << ") " << edges[paths1[numv - 1][i]].end1 << 
      " " << edges[paths1[numv - 1][i]].end2 << 
      " " << edges[paths1[numv - 1][i]].cost << 
      " tot " << totc << endl;
      nedges++;
      //tup = costQRMSE(edges[paths1[numv-1][i]].end1, edges[paths1[numv-1][i]].end2);
      tup = (this->*pntCost)(edges[paths1[numv-1][i]].end1, edges[paths1[numv-1][i]].end2);
      sol.push_back(tup);
   }
   cout << "F&B (Bellman-Ford) "<< dsName << " cost: " << std::setprecision(5) << totc << " n_brk " << nedges-1 << " t.cpu " << ttot << endl;
   return sol;
}

// ricostrusce la soluzione di bellman ford, DFS all'arovescia dalla fine
void FnBsegmentation::reconstructBF(vector<double> costs, vector<Edge>& edges, int numv, int maxNumEdges)
{  int i,j,v;
   int tcurr, end1, depth;;
   double minCost, edgeCost, c=0; // the incremental cost of the stack

   vector<bool> discovered(numv+1,false);
   vector<int>  preds;

   // Push the starting (i.e., end) vertex onto the stack
   v = edges[edges.size() - 1].end2+1;
   costs[0]=0;
   minCost = costs[numv-1];
   costs.push_back(minCost); // to initialize backward recursion

   //for(i=0;i<edges.size();i++)
   //   if(edges[i].end2==193)
   //      cout << i << "," << edges[i].end1<<"," << edges[i].end2<<"," << edges[i].cost<<endl;
}

// scrive una soluzione su file csv
int FnBsegmentation::writeSolCsv(vector<tuple<int, int, double, double, double>> sol, string fileName) 
{  int i;
   ofstream outFile(fileName); // Open file for writing

   if (!outFile) {
      cout << "Error opening file!" << endl;
      return 1;
   }

   outFile << "t1,t2,m,q,cost" << costName << endl;
   for (i=0;i<sol.size();i++) 
   {
      outFile << get<0>(sol[i]) << "," << 
         get<1>(sol[i]) << "," << 
         get<2>(sol[i]) << "," << 
         get<3>(sol[i]) <<"," << 
         get<4>(sol[i]) << endl;
   }

   outFile.close(); // Close the file
   cout << "Solution written to " << fileName << endl;
   return 0;
}

// cost as quasi RMSE
tuple<int, int, double, double, double> FnBsegmentation::costQRMSE(int t1, int t2)
{  int i, n;
   double m, q, r, sumres2 = 0, sumchi = 0;
   vector<int> x;
   vector<double> y;
   vector<double> ypred, residuals;

   n = t2-t1;
   for (i = t1; i < t2; i++)
   {  x.push_back(i);
      y.push_back(Y[i]);
   }

   tie(m, q) = linearRegression(x, y);
   for (i = 0; i < n; i++)
   {
      ypred.push_back(m * x[i] + q);
      r = y[i] - ypred[i];
      residuals.push_back(r);
      sumres2 += r * r;
   }
   double costQRMSE = sumres2 / sqrt(n);
   return { t1, t2, m, q, costQRMSE };
}

/// Compute AIC for a linear model yhat = m*x + q given y and (optionally) x.
/// If x is null, x[i] is assumed to be i (0,1,2,...).
/// The returned AIC is n*log(RSS/n) + 2*k, with k=3 (slope, intercept, sigma^2).
/// Set includeConstant=true to add n*log(2*pi) + n (software-dependent constant).
tuple<int, int, double, double, double> FnBsegmentation::costAIC(int t1, int t2)
{  int i, n;
   double m, q, rss=0;
   vector<int> x;
   vector<double> y;
   vector<double> ypred, residuals;
   bool includeConstant = false;  // per la formula AIC estesa

   n = t2-t1;
   for (i = t1; i < t2; i++)
   {  x.push_back(i);
      y.push_back(Y[i]);
   }

   tie(m, q) = linearRegression(x, y);

   for (i = 0; i < n; i++)
   {  double yhat = m * x[i] + q;
      double e = y[i] - yhat;
      rss += e * e;
   }

   // MLE of sigma^2 is RSS/n
   double mse = rss / n;

   // Number of estimated parameters: slope, intercept, and error variance sigma^2
   const int k = 3;

   // AIC without the constant (commonly used for comparison across models fit to the same data)
   double aic = n * log(mse) + 2 * k;

   // Optional: add the software-dependent constant n*log(2*pi) + n
   if (includeConstant)
      aic += n * log(2.0 * 3.141592) + n;

   // here it becomes AICc
   if(n<40)
      aic = aic + (2*k*k+2*k)/(n-k+1);

   return { t1, t2, m, q, aic};
}

// OLS line through vector of points
tuple<double, double> FnBsegmentation::linearRegression(vector<int> x, vector<double> y)
{
   int n, i;
   double sum_x = 0, sum_x2 = 0, sum_y = 0, sum_xy = 0, m, q;

   n = x.size();

   for (i = 0; i < n; i++)
   {
      sum_x  = sum_x  + x[i];
      sum_x2 = sum_x2 + x[i] * x[i];
      sum_y  = sum_y  + y[i];
      sum_xy = sum_xy + x[i] * y[i];
   }

   m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
   q = (sum_y - m * sum_x) / n;

   return { m,q };
}
