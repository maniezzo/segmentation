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
extern std::string global;    // tipo di vincolo globale, serie da ottimizzare
extern int maxIter;      // max nunm of lagrangian iterations
extern int maxTime;      // max secs of lagrangian runs
extern int nMaxSegm;     // max num segments
extern int minlength;    // min segment length
extern int firstRow;     // prima istanza in M3_4_sample da ottimizzare
extern int lastRow;      // prima istanza in M3_4_sample ESCLUSA
extern char solver;
extern std::vector<std::vector<int>> rowids, colids;  // compression of indices, by row and by col
