#include "global.h"
#include "FnBsegmentation.h"
#include "json.h"

// dataFiles = { "test","BTC-USD","IBatt_Min","PTemp_C_Avg","PTemp_C_Max","V_in_chg_Avg","Vapor_Pressure_Avg","ThermTemp1_Avg","WTemp_C2_Avg","SDI_Temp_1m","WS_ms_Avg","WTemp_C1_Avg","Vapor_Pressure_Avg_2","new507" };

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

   baseDir   = JSV["basedir"].ToString();
   dsName    = JSV["dsName"].ToString();
   maxNumEdges = JSV["maxNumEdges"];
   delta     = JSV["delta"];
   minLength = JSV["minLength"];
   maxLength = JSV["maxLength"];
   maxcpu    = JSV["maxcpu"];
   idcost    = JSV["idcost"];
   isVerbose = JSV["isVerbose"];
   cout << baseDir << endl;
   cout << dsName << endl;
}

// legge l'istanza
int readData(string dataFileName, vector<int>& X, vector<double>& Y)
{
   int i, cont, id;
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
      id   = stoi(elem[0]);
      X.push_back(id);
      d    = stod(elem[1]);
      Y.push_back(d); // i valori della serie
l0:      cont++;
      }
      f.close();
      n = Y.size();  // number of input records
   }
   else cout << "Cannot open dataset input file\n";
   return n;
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

int main()
{  int i, j, idcost, cont;
   vector<tuple<int, int, double, double, double>> lstOLS;

   FnBsegmentation FnB;

   std::cout << std::fixed;
   readConfig();
   zub = DBL_MAX;

   int    idDataSet = 0;
   string dataFile        = baseDir + dsName + ".csv";
   string segmentFileName = baseDir + dsName + "_runs.csv";
   vector<int> X;
   n = readData(dataFile,X,Y);
   FnB.run_FnB();
}
