#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <tuple>
#include <stack>
#include <queue>     // for min heap at each node
#include <functional>// for std::greater, sorting of the heaps (min)
#include <windows.h> // GetModuleFileName, for ExePath
#include <numeric>   // accumulate
#include <algorithm> // for_each
#include <iomanip>   // setprecision
#include <map>       // forward and backward nodes
#include <limits>    // needed in maps
#include <algorithm> // max, min
#include <ctime>     // clock_t

using namespace std;

const double EPS = 0.00001;

extern string  baseDir;
extern string  dsName;       // dataset name (file col .csv)
extern int     maxNumEdges;  // max number if segments
extern clock_t tstart, tend;
extern double  ttot,topt;
extern int n;                // num of points of the data series
extern int delta;            // width of the beam
extern int minLength;        // minimal segnment length
extern int maxLength;        // maximal segnment length
extern int maxcpu;           // max running time in seconds
extern int numMatch;         // number of partial solutions matching
extern int idcost;           // id (0 based) of the cost function
extern vector<double> Y;     // the dataseries to model
extern double zub;
extern double zlb;
extern bool isLB;            // whether to compute the lower bound at each iteration
extern bool isVerbose;
