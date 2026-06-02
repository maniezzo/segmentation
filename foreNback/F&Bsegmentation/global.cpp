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
int      idcost      = 0;           // id (0 based) of the cost function
double   zub         = 0.0;
double   zlb         = 0.0;
double   ttot        = 0.0;
double   topt        = 0.0;
double   tprec       = 0.0;
bool     isLB        = false;
bool     isVerbose   = true;
clock_t  tstart, tend;
vector<double> Y;
