#pragma once
#include "global.h"
#include <ilcplex/cplex.h>

void goCPLEX();
int populatebyrow(CPXENVptr env, CPXLPptr lp, vector<double> y, vector<tuple<int, int, double, double, double>> lstOLS);
