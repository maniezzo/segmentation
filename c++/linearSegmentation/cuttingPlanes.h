#pragma once
#include "global.h"
#include <ilcplex/cplex.h>

vector<double> goCutPlanes(vector<double> y,
   vector<tuple<int, int, double, double, double>> lstOLS,
   int nMaxSegm, string cons);
vector<int> separateUpDownCuts(const vector<double>& x);
