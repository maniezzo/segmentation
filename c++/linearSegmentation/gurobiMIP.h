#pragma once
#include <gurobi_c++.h>
#include "global.h"

int populateGurobiByRow(GRBModel& model,
   const vector<double>& y,
   const vector<tuple<int,int,double,double,double>>& lstOLS,
   vector<GRBVar>& xVars,
   vector<GRBConstr>& constrs);
vector<double> goGurobi(vector<double> y, vector<tuple<int, int, double, double, double>> lstOLS);