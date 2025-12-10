#include "FnBsegmentation.h"
#include "Stage.h"

void FnBsegmentation::run_FnB()
{  int i;
   Stage N;
   N.mainNode();
   Fstage.resize(n);
   for(i=1;i<n;i++) Fstage[i].insert(0, DBL_MAX);
   Fstage[0].insert(0, 0);
   Bstage.resize(n);
   for(i=0;i<n-1;i++) Bstage[i].insert(0, DBL_MAX);
   Bstage[n-1].insert(0, 0);

   forward();
   //backward();
   /*
   readSegments(segmentFileName,lstOLS);
   if(get<0>(lstOLS[0]) != 0) cout << ">>>> ERROR <<<< dataseries not starting at t=0. Disposable results." << endl;

   DAG_SSSP(lstOLS);
   run_BF(lstOLS, maxNumEdges);
   */
}

// forward pass
void FnBsegmentation::forward()
{  int i,j,t,k;
   Stage seed;
   double z,lb;  // lower bound to completion

   for(i=0;i<n;i++)
   {  for(k=0;k<delta;k++)
      {  z  = Fstage[i].queryMinCost(0).second;
         lb = Bstage[i].queryMinCost(0).second;

         if(z + lb >= zub)
         {  nFathomed++;
            continue;
         }

         if(seed.nSegm < maxNumEdges)
            for(t=seed.t2+minLength;t<n;t++)
               generateFoffspring(seed.t2,t,seed.nSegm,seed.cost);
      }
   }
   return;
}

// backward pass
void FnBsegmentation::backward()
{  int i,j,t,k;
   NodeST seed;
   double lb;  // lower bound to completion

   for (i=n-1;i<<n>=0;i--)
   {  for(k=0;k<delta;k++)
      {  if(BminHeaps[i].empty())
            continue;
         seed = BminHeaps[i].top();
         BminHeaps[i].pop();   // removes after assigning

         if(i==1 || FminHeaps[i-1].empty())
            lb = 0;
         else
            lb = FminHeaps[i-1].top().cost;

         if(seed.cost + lb >= zub)
         {  nFathomed++;
            continue;
         }

         if(seed.nSegm < maxNumEdges)
            for (t=seed.t2-minLength;t>=0;t--)
               generateBoffspring(seed.t2,t,seed.nSegm,seed.cost);
      }
   }
   return;
}

// forward offspring generation
void FnBsegmentation::generateFoffspring(int t1, int t2, int nSegm, double cost)
{  int i;
   double c,m,q,z;

   tuple<int,int,double,double,double> res = costQRMSE(t1, t2);
   z = cost+get<4>(res);
   if (t2==(n-1) && z<zub)
   {  zub = z;
      cout << "F) New zub: "<< zub << endl;
   }
   NodeST nd = {t1,t2,z,nSegm+1};
   Fstage[t2].push(nd);
   //cout << "t1=" << t1 << " t2=" << t2 << " nSegm="<< nSegm << endl;

   return;
}

// backward offspring generation
void FnBsegmentation::generateBoffspring(int t2, int t1, int nSegm, double cost)
{  int i;
   double c,m,q,z;

   tuple<int,int,double,double,double> res = costQRMSE(t1, t2);
   z = cost+get<4>(res);
   if (t2==(n-1) && z<zub)
   {  zub = z;
      cout << "B) New zub: "<< zub << endl;
   }
   NodeST nd = {t2,t1,z,nSegm+1};
   BminHeaps[t2].push(nd);
   //cout << "t1=" << t1 << " t2=" << t2 << " nSegm="<< nSegm << endl;

   return;
}

// single source on a DAG
void FnBsegmentation::DAG_SSSP(vector<tuple<int, int, double, double, double>> lstOLS)
{  int i,j,n,t,currInit,maxt,maxstart;
   double c;
   vector<int> initSegm;   // indice in lstOLS inizio indici segmenti con inizio al tempo i
   vector<int> sol;

   n = lstOLS.size();
   currInit = -1;
   tstart = clock();

   for (i = 0; i < n; i++)
      if(get<0>(lstOLS[i]) > currInit)
      {  initSegm.push_back(i);
         currInit = get<0>(lstOLS[i]);
      }

   initSegm.push_back(i);
   maxt = get<1>(lstOLS[n-1]);

   vector<double> mincost(maxt+1,DBL_MAX);  // min cost for covering up to time t
   vector<int> minsegm(maxt+1,-1); // id last segment producing cost mincost[t]
   maxstart = get<0>(lstOLS[n-1]);

   for (t=0;t<=maxstart;t++)
      for(i=initSegm[t];i<initSegm[t+1];i++)
      {  j = get<1>(lstOLS[i]);
         c = (t>0 ? mincost[t-1] : 0) + get<4>(lstOLS[i]);
         if (mincost[j] > c)
         {  mincost[j] = c;
            minsegm[j] = i;
         }
      }
   tend = clock();
   ttot = (tend - tstart) / CLOCKS_PER_SEC;

   sol = reconstructSolution(lstOLS,minsegm,maxt);
   cout << "F&B (DAG) " << dsName << " cost: " << std::setprecision(5) << mincost[maxt] << " n_brk " << sol.size()-1 << " t.cpu " << ttot << endl;
}

// Bellman-Ford algorithm with bounded number of edges
void FnBsegmentation::bellmanFord(vector<Edge>& edges, int numv, int maxNumEdges) 
{  int i,j,u,v,t;
   double w,du;
   vector<double> cost0(numv, DBL_MAX);
   vector<double> cost1(numv, DBL_MAX); // cost matrix after iteration update
   vector<vector<int>> paths0(numv); // final paths
   vector<vector<int>> paths1(numv); // final paths after iteration update
   vector<int> prev(numv, -1);
   vector<int> minsegm(numv, -1);

   tstart = clock();
   for(i=0;i<numv;i++)
      paths0[i] = vector<int>();
   cost0[0] = 0;

   // stages
   for (i = 0; i < min(maxNumEdges,numv-1); ++i)
   {  for (j=0;j<edges.size();j++)
      {  u = edges[j].end1;
         v = edges[j].end2;
         w = edges[j].cost;

         du = 0;
         if(u>0) du = cost0[u-1]; // il nuovo segm parte 1 dopo la fine del precedente
         if (cost0[u] != DBL_MAX && (du + w) < cost1[v])
         {  cost1[v] = du + w;
            prev[v] = u;
            minsegm[v]  = edges[j].segm;
            paths1[v] = (u==0 ? paths0[u] : paths0[u-1]);
            paths1[v].push_back(j);
         }
      }
      cost0 = cost1; // cost1 will be the cost0 at next iteration
      paths0 = paths1;
   }
   tend = clock();
   ttot = (tend - tstart) / CLOCKS_PER_SEC;

   // Check for negative weight cycles useless in DAG

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
   }
   cout << "F&B (Bellman-Ford) "<< dsName << " cost: " << std::setprecision(5) << totc << " n_brk " << nedges-1 << " t.cpu " << ttot << endl;
}

// imposta poi lancia bellman fors
int FnBsegmentation::run_BF(vector<tuple<int, int, double, double, double>> lstOLS, int maxNumEdges)
{  int i;
   int numEdges = lstOLS.size();
   int numv = get<1>(lstOLS[numEdges-1])+1; // partono da 0
   vector<Edge> edges;

   cout << "For each edge:" << endl;
   for (int i = 0; i < numEdges; ++i) {
      Edge edge;
      edge.end1 = get<0>(lstOLS[i]);
      edge.end2 = get<1>(lstOLS[i]);
      edge.segm = i;
      edge.cost = get<4>(lstOLS[i]);
      edges.push_back(edge);
   }

   bellmanFord(edges, numv, maxNumEdges);

   return 0;
}

// ricostruisce la soluzione DAG
vector<int> FnBsegmentation::reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, vector<int> minsegm, int maxt)
{  int i,j,t;
   double sum = 0;
   vector<int> sol;

   t = maxt;
   while(t>0)
   {  i = minsegm[t];
      sum += get<4>(lstOLS[i]);  // costo integrale
      sol.push_back(i);
      cout << "Segmento " << i << " " << get<0>(lstOLS[i]) << "-" << get<1>(lstOLS[i]) << " costo " << get<4>(lstOLS[i]) << endl;
      j = get<1>(lstOLS[i]) - get<0>(lstOLS[i]) +1;
      t -= j;
   }
   cout << "Costo complessivo " << sum << endl;
   return sol;
}

// OUTDATED, UNUSED - ricostrusce la soluzione di bellman ford, DFS all'arovescia dalla fine
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
      sum_x  = sum_x + x[i];
      sum_x2 = sum_x2 + x[i] * x[i];
      sum_y  = sum_y + y[i];
      sum_xy = sum_xy + x[i] * y[i];
   }

   m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
   q = (sum_y - m * sum_x) / n;

   return { m,q };
}
