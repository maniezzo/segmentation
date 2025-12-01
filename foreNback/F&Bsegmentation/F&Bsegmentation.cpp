#include "F&Bsegmentation.h"
#include "json.h"

// dataFiles = { "test","BTC-USD","IBatt_Min","PTemp_C_Avg","PTemp_C_Max","V_in_chg_Avg","Vapor_Pressure_Avg","ThermTemp1_Avg","WTemp_C2_Avg","SDI_Temp_1m","WS_ms_Avg","WTemp_C1_Avg","Vapor_Pressure_Avg_2","new507" };

int main()
{  int           i, j, n, idcost, cont;
   int           cur_numrows, cur_numcols;
   vector<tuple<int, int, double, double, double>> lstOLS;

   std::cout << std::fixed;
   readConfig();

   int idDataSet = 0;
   string dataFile        = baseDir + dsName + ".csv";
   string segmentFileName = baseDir + dsName + "_runs.csv";
   readSegments(segmentFileName,lstOLS);
   if(get<0>(lstOLS[0]) != 0) cout << ">>>> ERROR <<<< dataseries not starting at t=0. Disposable results." << endl;

   DAG_SSSP(lstOLS);
   run_BF(lstOLS, maxNumEdges);
}

// single source on a DAG
void DAG_SSSP(vector<tuple<int, int, double, double, double>> lstOLS)
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
void bellmanFord(vector<Edge>& edges, int numv, int maxNumEdges) 
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
   cout << "F&B (nbrk) "<< dsName << " cost: " << std::setprecision(5) << totc << " n_brk " << nedges-1 << " t.cpu " << ttot << endl;
}

int run_BF(vector<tuple<int, int, double, double, double>> lstOLS, int maxNumEdges)
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
vector<int> reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, vector<int> minsegm, int maxt)
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
void reconstructBF(vector<double> costs, vector<Edge>& edges, int numv, int maxNumEdges)
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

// legge i segmenti precalcolati
void readSegments(string segmentFileName, vector<tuple<int, int, double, double, double>> & lstOLS)
{  int i,j,n=0,cont=0;
   int id,low,hi;
   double m,q,cost;
   string line;
   vector<string> elem;
   tuple<int, int, double, double, double> segm;
   ifstream fs;

   cout << "Opening segment file " << segmentFileName << endl;
   fs.open(segmentFileName);
   if (fs.is_open())
   {
      getline(fs, line);  // headers
      cout << line << endl;
      elem = split(line, ',');

      while (getline(fs, line))
      {
         cont = 0;
         elem = split(line, ',');
         id   = stoi(elem[0]);
         low  = stoi(elem[1]);
         hi   = stoi(elem[2]);
         m    = stod(elem[3]);
         q    = stod(elem[4]);
         cost = stod(elem[5]);
         segm = make_tuple(low, hi, m, q, cost);
         lstOLS.push_back(segm);
         cont++;
      }
      fs.close();
      n = lstOLS.size();  // number of input records
      cout << "Read " << n << " segments" << endl;
   }
   else cout << "Cannot open segment input file\n";
}

// split di una stringa in un array di elementi delimitati da separatori
vector<string> split(string str, char sep)
{
   vector<string> tokens;
   size_t start;
   size_t end = 0;
   while ((start = str.find_first_not_of(sep, end)) != std::string::npos) {
      end = str.find(sep, start);
      tokens.push_back(str.substr(start, end - start));
   }
   return tokens;
}

int readData(string dataFileName, vector<int>& X, vector<double>& Y)
{
   int i, cont, id, n = 0;
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
      {
         cont = 0;
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

// OLS line through vector of points
tuple<double, double> linearRegression(vector<int> x, vector<double> y)
{
   int n, i;
   double sum_x = 0, sum_x2 = 0, sum_y = 0, sum_xy = 0, m, q;

   n = x.size();

   for (i = 0; i < n; i++)
   {
      sum_x = sum_x + x[i];
      sum_x2 = sum_x2 + x[i] * x[i];
      sum_y = sum_y + y[i];
      sum_xy = sum_xy + x[i] * y[i];
   }

   m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
   q = (sum_y - m * sum_x) / n;

   return { m,q };
}

// datafile etc.
void readConfig()
{  int i,j;
   string line;

   cout << "Running from " << exePath() << endl;

   ifstream fconf("config.json");
   stringstream buffer;
   buffer << fconf.rdbuf();
   line = buffer.str();
   json::Value JSV = json::Deserialize(line);

   baseDir = JSV["basedir"].ToString();
   dsName  = JSV["dsName"].ToString();
   maxNumEdges = JSV["maxNumEdges"];
   cout << baseDir << endl;
   cout << dsName << endl;
}

// trova il path del direttorio da cui si e' lanciato l'eseguibile
string exePath()
{
   wchar_t buffer[MAX_PATH];
   GetModuleFileName(NULL, buffer, MAX_PATH);
   wstring ws(buffer);
   string s = string(ws.begin(), ws.end());
   string::size_type pos = s.find_last_of("\\/");
   return s.substr(0, pos);
}
