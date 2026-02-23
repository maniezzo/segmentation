#pragma once
#include "global.h"
#include <ilcplex/cplex.h>

vector<double> goCPLEX(vector<double> y, vector<tuple<int, int, double, double, double>> lstOLS, int ntot);
int populatebyrow(CPXENVptr env, CPXLPptr lp, vector<double> y, vector<tuple<int, int, double, double, double>> lstOLS);
