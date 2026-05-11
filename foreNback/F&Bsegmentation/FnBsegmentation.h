#pragma once
#include "global.h"
#include "Stage.h"
#include "DPtable.h"
#include <memory>     // For std::unique_ptr

class FnBsegmentation
{
private:
   int nFathomed;        // eliminated by the bound
   string costName;      // name of the cost function

   struct Edge { int end1, end2, segm; double cost; }; // an edge in bellman ford

   unique_ptr<DPtable> Fstage, Bstage; // expansion stages, unexpanded nodes
   vector<int> changepoints; // final changepoints

   bool forward();
   bool backward();
   bool generateFoffspring(int t1, int t2, int nSegm, double cost, vector<int> lstPoints);
   bool generateBoffspring(int t1, int t2, int nSegm, double cost, vector<int> lstPoints);
   bool match(bool isForward, int i, int numSegm, double z);
   double computeLB();
   void reconstructFnBsolution();
   tuple<double, double> linearRegression(vector<int> x, vector<double> y);
   tuple<int, int, double, double, double> costQRMSE(int t1, int t2);
   tuple<int, int, double, double, double> costAIC(int t1, int t2);
   void DAG_SSSP();
   vector<tuple<int, int, double, double, double>> reconstructDAGsolution(vector<tuple<int, int, double, double, double>> lstOLS, int);
   int run_BF();
   void reconstructBF(vector<double> costs, vector<Edge>& edges, int numv, int maxNumEdges);
   vector<tuple<int, int, double, double, double>> bellmanFord(vector<Edge>& edges, int numv, int maxNumEdges);
   int writeSolCsv(vector<tuple<int, int, double, double, double>> lstOLS, string fileName);
   tuple<int, int, double, double, double> (FnBsegmentation::*pntCost)(int, int) = nullptr; // pointer to cost function

public:
   void run_FnB();
};

