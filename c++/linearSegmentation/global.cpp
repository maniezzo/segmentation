#include "global.h"
string baseDir = "";
string dsName = "";
char solver = 'h';
vector<vector<int>> colids;
vector<vector<int>> rowids;
int maxIter = 0;
int maxTime = 0;
int nMaxSegm    = 0;  // max num segments
int minlength;    // min segment length
int firstRow= 0;  // prima istanza in M3_4_sample da ottimizzare
int lastRow = 1;  // prima istanza in M3_4_sample ESCLUSA
