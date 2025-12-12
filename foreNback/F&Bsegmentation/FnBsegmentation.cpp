#include "FnBsegmentation.h"
#include "Stage.h"

// n number of points, maxt maximum time (=n-1)
void FnBsegmentation::run_FnB()
{  int i;
   bool isImprovedF = true, isImprovedB = true;;

   // optimal solution with no constraint on the number of arcs (but with minLength)
   DAG_SSSP();

   // optimal solution with one constraint on the number of arcs and with minLength
   run_BF();

   Stage N;
   N.mainNode();
   Fstage.resize(n);
   Fexpanded.resize(n);
   Fstage[0].insert(0, 0);
   Bstage.resize(n);
   Bexpanded.resize(n);
   Bstage[n-1].insert(0, 0);

   while(isImprovedF || isImprovedB)
   {  isImprovedF = forward();
      isImprovedB = backward();
   }
}

// forward pass
bool FnBsegmentation::forward()
{  int i,j,t,k,nSegm;
   Stage seed;
   double z,lb;  // lower bound to completion
   bool isImproved = false;

   for(i=0;i<n;i++)
   {  if(i>0 && i<minLength) continue;  // non ho segmenti più corti di minLength

      for(k=0;k<delta;k++) // beam width
      {  nSegm = Fstage[i].queryMinCost(maxNumEdges).second;  // num of segments up to stage i
         z     = Fstage[i].queryMinCost(maxNumEdges).first;   // cost up to stage i

         if(Fstage[i].isEmpty())
            continue;
         // remove from unexpanded and add to expanded
         Fexpanded[i].insert(nSegm, z);
         auto res = Fstage[i].popMinCost(maxNumEdges);
         if (res.first!=z||res.second!=nSegm)
            cout << ">>>> ERROR <<<< popping from Fstage inconsistent." << endl;

         if (i<(n-minLength))
            lb = Bstage[i+1].queryMinCost(maxNumEdges).first;
         else
            lb = 0;

         if(z + lb >= zub)
         {  nFathomed++;
            continue;
         }

         if(nSegm < maxNumEdges)
            for(t=i+minLength;t<n;t++)
               isImproved = generateFoffspring(i,t,nSegm,z);
      }
   }
   return isImproved;
}

// backward pass
bool FnBsegmentation::backward()
{  int i,j,t,k,nSegm;
   Stage seed;
   double z,lb;  // lower bound to completion
   bool isImproved = false;

   int maxt = n-1;
   for(i=maxt;i>=0;i--)
   {  if (i<maxt && (maxt-i)<minLength) continue;  // non ho segmenti più corti di minLength

      for(k=0;k<delta;k++) // beam width
      {  nSegm = Bstage[i].queryMinCost(maxNumEdges).second;  // num of segments up to stage i
         z     = Bstage[i].queryMinCost(maxNumEdges).first;   // cost up to stage i

         if(Bstage[i].isEmpty())
            continue;
         // remove from unexpanded and add to expanded
         Bexpanded[i].insert(nSegm, z);
         auto res = Bstage[i].popMinCost(maxNumEdges);
         if (res.first!=z||res.second!=nSegm)
            cout << ">>>> ERROR <<<< popping from Fstage inconsistent." << endl;

         if (i>minLength)
            lb = Fstage[i-1].queryMinCost(maxNumEdges).first;
         else
            lb = 0;

         if(z + lb >= zub)
         {  nFathomed++;
            continue;
         }

         if(nSegm < maxNumEdges)
            for (t=i-minLength;t>=0;t--)
               isImproved = generateBoffspring(i,t,nSegm,z); // backward, t<i
      }
   }
   return isImproved;
}

// forward offspring generation
bool FnBsegmentation::generateFoffspring(int t1, int t2, int nSegm, double cost)
{  int i;
   double c,m,q,z;
   bool isImproved = false;

   tuple<int,int,double,double,double> res = costQRMSE(t1, t2);
   z = cost+get<4>(res);
   if (t2==(n-1) && z<zub)
   {  zub = z;
      cout << "F) New zub: "<< zub << endl;
      isImproved = true;
   }

   if(cost+z < zub)
      Fstage[t2].insert(nSegm+1, cost+z);
   else
      nFathomed++;
   //cout << "t1=" << t1 << " t2=" << t2 << " nSegm="<< nSegm+1 << " cost " << cost+z << endl;

   return isImproved;
}

// backward offspring generation
bool FnBsegmentation::generateBoffspring(int t2, int t1, int nSegm, double cost)
{  int i;
   double c,m,q,z;
   bool isImproved = false;

   tuple<int,int,double,double,double> res = costQRMSE(t1, t2);
   z = cost+get<4>(res);
   if (t1==0 && z<zub)
   {  zub = z;
      cout << "B) New zub: "<< zub << endl;
      isImproved = true;
   }

   if(cost+z < zub)
      Bstage[t1].insert(nSegm+1, cost+z);
   else
      nFathomed++;
   //cout << "t1=" << t1 << " t2=" << t2 << " nSegm="<< nSegm+1 << " cost " << cost+z << endl;

   return isImproved;
}

// matches the current partial solution against one from the opposite direction
bool FnBsegmentation::match(bool isForward, int i, int numSegm, double z)
{  double lb; // cost of completion
   bool hasMatch = false;

   if (isForward)
   {  lb = Bexpanded[i+1].queryMinCost(maxNumEdges-numSegm).first;
      if(lb==0) goto l0; // no feasible expansion
      hasMatch = true;
      if(z + lb < zub)
      {  // new incumbent
         zub = z + lb;
         cout << "F) New zub: "<< zub << endl;
      }
   }
   else
   {  lb = Fexpanded[i-1].queryMinCost(maxNumEdges-numSegm).first;
      if(lb==0) goto l0; // no feasible expansion
      hasMatch = true;
      if(z + lb < zub)
      {  // new incumbent
         zub = z + lb;
         cout << "B) New zub: "<< zub << endl;
      }
   }
l0:return hasMatch;
}

// single source on a DAG. n number of points, maxt maximum time (=n-1)
void FnBsegmentation::DAG_SSSP()
{  int i,j,t,currInit,maxt,maxstart;
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
   {  for(j=t+minLength;j<=maxt;j++)
      {  if(t>0 && t<minLength) continue;
         tup = costQRMSE(t,j);  // cosi' segmenti attaccati, se staccati da t a j-1
         c   = get<4>(tup);
         if (mincost[j]==DBL_MAX || mincost[j]>(mincost[t]+c))
         {  mincost[j] = mincost[t] + c;
            lstOLS[j]  = tup;
         }
      }
   }
   tend = clock();
   ttot = (tend - tstart) / CLOCKS_PER_SEC;

   sol = reconstructSolution(lstOLS,maxt);
   writeSolCsv(sol,"test_sol.csv");
   cout << "F&B (DAG) " << dsName << " cost: " << std::setprecision(5) << mincost[maxt] << " n_brk " << sol.size()-1 << " t.cpu " << ttot << endl;
}

// ricostruisce la soluzione DAG
vector<tuple<int, int, double, double, double>> FnBsegmentation::reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, int maxt)
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
{  int cont;
   int numEdges = 0;
   int numv     = n; // partono da 0
   vector<Edge> edges;
   vector<tuple<int, int, double, double, double>> lstOLS;
   vector<tuple<int, int, double, double, double>> sol;

   cout << "For each edge:" << endl;
   cont=0;
   for (int t1 = 0; t1 < n-minLength; ++t1) 
   {  if(t1>0 && t1<minLength) continue;
      for (int t2 = t1+minLength; t2 < n; ++t2) 
      {  if(t2<n-1 && t2 > n-minLength) continue;
         Edge edge;
         edge.end1 = t1;
         edge.end2 = t2;
         lstOLS.push_back(costQRMSE(t1,t2));
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
      tup = costQRMSE(edges[paths1[numv-1][i]].end1, edges[paths1[numv-1][i]].end2);
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

   outFile << "t1,t2,m,q,cost" << endl;
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
