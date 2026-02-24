#include "hexalyMIP.h"

// Build the model (Hexaly)
void populateHexalyModel(  HxModel& model,
                           const vector<double>& y,
                           const vector<tuple<int,int,double,double,double>>& lstOLS,
                           vector<HxExpression>& xVars,
                           vector<HxExpression>& constrs,
                           int ntot,
                           bool relaxLP,          // true => x in [0,1] float relaxation, false => x is boolean
                           bool partitioning       // true => ==1, false => <=1 (covering / packing style)
                        )
{  const int n = (int)lstOLS.size();
   const int m = (int)y.size();

   xVars.clear();
   xVars.reserve(n);
   constrs.clear();
   constrs.reserve(m + 1);

   // ---- Variables ----
   // Gurobi: addVar(0,1,obj, CONTINUOUS/BINARY)
   // Hexaly: create decisions, then build objective separately.
   for (int j = 0; j < n; ++j) {
      HxExpression xj = relaxLP ? model.floatVar(0.0, 1.0) : model.boolVar(); // boolVar is {0,1}
      // Optional: name (useful for debugging / exporting)
      xj.setName(("x" + to_string(j)).c_str());
      xVars.push_back(xj);
   }

   // ---- Cover/partition constraints: for each i, sum_{j covers i} x_j == 1 (or <= 1) ----
   for (int i = 0; i < m; ++i) 
   {  HxExpression sum = model.createExpression(O_Sum); // dynamic sum via addOperand

      for (int j = 0; j < n; ++j) 
      {  int start = get<0>(lstOLS[j]);
         int end   = get<1>(lstOLS[j]);
         if (i >= start && i <= end) 
            sum.addOperand(xVars[j]);
      }

      // Build boolean constraint expression, then add it
      HxExpression c =
         partitioning
         ? model.eq(sum, 1)     // sum == 1
         : model.leq(sum, 1);   // sum <= 1

      c.setName(("c" + to_string(i)).c_str());
      model.constraint(c);          // boolean expression must be true
      constrs.push_back(c);
   }

   // ---- Cardinality constraint: sum_j x_j <= ntot ----
   HxExpression card = model.createExpression(O_Sum);
   for (int j = 0; j < n; ++j) card.addOperand(xVars[j]);

   HxExpression ntotC = model.leq(card, ntot);
   ntotC.setName("ntot");
   model.constraint(ntotC);
   constrs.push_back(ntotC);

   // ---- Objective: minimize sum_j cost_j * x_j ----
   HxExpression obj = model.createExpression(O_Sum);
   for (int j = 0; j < n; ++j) 
   {  double cost = get<4>(lstOLS[j]);
      // product(cost, x_j) then sum
      obj.addOperand(model.prod(cost, xVars[j]));
   }
   model.minimize(obj);

   // Must close before solve
   model.close();
}

// Main function
vector<double> goHexaly(const vector<double>& y,
                        const vector<tuple<int, int, double, double, double>>& lstOLS,
                        int ntot,
                        int timeLimitSeconds,   // optional
                        bool relaxLP,        // false => MIP-style (boolean) solve; true => [0,1] relaxation
                        bool partitioning    // true => ==1 ; false => <=1
                     )
{  clock_t tstart, truns, tMIP;
   vector<double> xnil;
   vector<double> x;

   try 
   {
      HexalyOptimizer optimizer;
      HxModel model = optimizer.getModel();

      // Similar to Gurobi OutputFlag: use verbosity 0/1/2
      optimizer.getParam().setVerbosity(1);
      optimizer.getParam().setTimeLimit(timeLimitSeconds);

      vector<HxExpression> xVars;
      vector<HxExpression> constrs;

      populateHexalyModel(model, y, lstOLS, xVars, constrs, ntot, relaxLP, partitioning);

      optimizer.solve();

      const int n = (int)xVars.size();
      x.resize(n);

      // Retrieve solution values
      for (int j = 0; j < n; ++j) 
         if (relaxLP) 
            x[j] = xVars[j].getDoubleValue(); // in [0,1]
         else 
            x[j] = (double)xVars[j].getValue(); // bool/int value 0 or 1

      // If you want objective value: model.getObjective(...) exists, but easiest is
      // to store 'obj' expression as a member and call obj.getDoubleValue()/getIntValue().
      // (Kept minimal here.)

      return x;
   }
   catch (const exception& e) 
   {  cout << "Hexaly exception: " << e.what() << endl;
      return {};
   }
}
