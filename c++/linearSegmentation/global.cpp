#include "global.h"
string baseDir = "";
string dsName = "";
string global = "";
char solver = 'h';
vector<vector<int>> colids;
vector<vector<int>> rowids;
int maxIter = 0;
int maxTime = 0;
int nMaxSegm    = 0;  // max num segments
int minlength;    // min segment length
int firstRow= 0;  // prima istanza in M3_4_sample da ottimizzare
int lastRow = 1;  // prima istanza in M3_4_sample ESCLUSA


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
