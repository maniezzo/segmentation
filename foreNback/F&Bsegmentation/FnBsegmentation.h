#pragma once
#include "global.h"
#include "Stage.h"

class FnBsegmentation
{
private:
   int nFathomed;        // eliminated by the bound

   struct Edge { int end1, end2, segm; double cost; }; // an edge in bellman ford

   struct NodeST      // an old node of the search tree
   {  int t1, t2;   // initial, final time point
      double cost;  // segment cumulative cost   (current included)
      int nSegm;    // number of segments so far (current included)

      // For min-heap based on cost
      bool operator>(const NodeST& pNode) const 
      {  return cost>pNode.cost;
      }
   };

   vector<Stage> Fstage, Bstage; // expansion stages


   void generateFoffspring(int t1, int t2, int nSegm, double cost);
   void generateBoffspring(int t1, int t2, int nSegm, double cost);
   tuple<double, double> linearRegression(vector<int> x, vector<double> y);
   tuple<int, int, double, double, double> costQRMSE(int t1, int t2);
   void DAG_SSSP(vector<tuple<int, int, double, double, double>> lstOLS);
   vector<int> reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, vector<int> minsegm, int);
   int run_BF(vector<tuple<int, int, double, double, double>> lstOLS, int maxNumEdges);
   void reconstructBF(vector<double> costs, vector<Edge>& edges, int numv, int maxNumEdges);
   void bellmanFord(vector<Edge>& edges, int numv, int maxNumEdges);
   void forward();
   void backward();

public:
   void run_FnB();
};

