#pragma once
#include "global.h"
#include "Stage.h"
#include "DPtable.h"
#include <memory>     // For std::unique_ptr
#include <cstddef>
#include <cassert>

class FnBsegmentation
{
private:
   int nFathomed;    // eliminated by the bound
   int numBF;        // number of back and forth iterations
   string costName;  // name of the cost function

   struct Edge { int end1, end2, segm; double cost; }; // an edge in bellman ford

   unique_ptr<DPtable> Fstage, Bstage; // expansion stages, unexpanded nodes
   vector<int> changepoints; // final changepoints

   vector<double> _arrOLS; // t1*n+t2 -> cost of the segment. Accesso come arrOLS via gli inline dopo
   // Helper: convert (i,j) to 1D index (row-major)
   inline std::size_t idx(std::size_t i, std::size_t j) const { return i * n + j;} // per accedere come fosse un array 2d
   // Matrix-style accessors
   inline double& arrOLS(std::size_t i, std::size_t j) {
      assert(i < n && j < n && "arrOLS: index out of bounds");
      return _arrOLS[idx(i, j)];
   }
   inline const double& arrOLS(std::size_t i, std::size_t j) const {
      assert(i < n && j < n && "arrOLS: index out of bounds");
      return _arrOLS[idx(i, j)];
   }

   void precompute_segm(); // precalcolo costi
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


   // precomputed matrix access: returns reference for read & write
   double& atOLS(std::size_t i, std::size_t j);
   const double& atOLS(std::size_t i, std::size_t j) const;

   // precomputed matrix: cleaner mat(i, j) syntax 
   double& operator()(std::size_t i, std::size_t j);
   const double& operator()(std::size_t i, std::size_t j) const;

public:
   void run_FnB();
};
