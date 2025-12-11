#pragma once
#include "global.h"
#include "Stage.h"

class FnBsegmentation
{
private:
   int nFathomed;        // eliminated by the bound

   struct Edge { int end1, end2, segm; double cost; }; // an edge in bellman ford

   vector<Stage> Fstage, Bstage; // expansion stages, unexpanded nodes
   vector<Stage> Fexpanded, Bexpanded; // expanded nodes


   void generateFoffspring(int t1, int t2, int nSegm, double cost);
   void generateBoffspring(int t1, int t2, int nSegm, double cost);
   tuple<double, double> linearRegression(vector<int> x, vector<double> y);
   tuple<int, int, double, double, double> costQRMSE(int t1, int t2);
   void DAG_SSSP();
   vector<tuple<int, int, double, double, double>> reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, int);
   int run_BF();
   void reconstructBF(vector<double> costs, vector<Edge>& edges, int numv, int maxNumEdges);
   void bellmanFord(vector<Edge>& edges, int numv, int maxNumEdges);
   void forward();
   void backward();
   int writeSolCsv(vector<tuple<int, int, double, double, double>> lstOLS, string fileName);

public:
   void run_FnB();
};

