#pragma once
#include <vector>
#include <string.h>
#include <iostream>
#include <sstream>
#include <time.h>
#include <tuple>


using namespace std;

extern std::string baseDir;
extern std::string dsName;    // dataset name (file col .csv)
extern int maxIter;      // max nunm of lagrangian iterations
extern int maxTime;      // max secs of lagrangian runs
extern std::vector<std::vector<int>> rowids, colids;  // compression of indices, by row and by col
