#pragma once
#include "global.h"
#include "c:/hexaly_14_5/include/optimizer/hexalyoptimizer.h"

using namespace hexaly;

vector<double> goHexaly(const vector<double>& y,
   const vector<tuple<int, int, double, double, double>>& lstOLS,
   int ntot,
   int timeLimitSeconds = 10,   // optional
   bool relaxLP = false,        // false => MIP-style (boolean) solve; true => [0,1] relaxation
   bool partitioning = true     // true => ==1 ; false => <=1
);
void populateHexalyModel(
   HxModel& model,
   const vector<double>& y,
   const vector<tuple<int,int,double,double,double>>& lstOLS,
   vector<HxExpression>& xVars,
   vector<HxExpression>& constrs,
   int ntot,
   bool relaxLP = false,          // true => x in [0,1] float relaxation, false => x is boolean
   bool partitioning = true       // true => ==1, false => <=1 (covering / packing style)
);
