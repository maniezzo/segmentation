#include "global.h"

// definitions and initializations
string   baseDir;
string   dsName;
int      maxNumEdges = 0; 
int      n           = 0;
int      delta       = 0;
int      minLength   = 0;
int      maxLength   = 0;
int      maxcpu      = 0;
int      numMatch    = 0;
double   zub         = 0.0;
double   ttot        = 0.0;
clock_t  tstart, tend;
vector<double> Y;
