#include "cuttingPlanes.h"


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

int readM3Data(string dataFileName, vector<int>& X, vector<double>& Y)
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
         id   = stoi(elem[0]);
         X.push_back(id);
         d    = stod(elem[1]);
         Y.push_back(d);
l0:      cont++;
   }
   f.close();
   n = Y.size();  // number of input records
}
else cout << "Cannot open dataset input file\n";
return n;
}


void goCutPlanes(string series, string cons)
{
   cout << "Fine cutting planes";
}