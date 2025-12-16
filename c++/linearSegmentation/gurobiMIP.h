#pragma once
#include "gurobi_c++.h"
#include "global.h"

int populateGurobiByRow(GRBModel& model,
   const vector<double>& y,
   const vector<tuple<int,int,double,double,double>>& lstOLS,
   vector<GRBVar>& xVars,
   vector<GRBConstr>& constrs);
int goGurobi();