#include "global.h"
string baseDir = "";
string dsName = "";
char solver = 'h';
vector<vector<int>> colids;
vector<vector<int>> rowids;
int maxIter = 0;
int maxTime = 0;
int ntot    = 0;  // max num segments
int minlength;    // min segment length
