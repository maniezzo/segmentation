#pragma once
#include "global.h"

// table of the dynamic programming recursion, it holds stage dat
class DPtable {
private:
   struct Cell {double z; vector<int> chpt;};
   vector<vector<Cell>> table;

public:
   // Constructor: initialize the table with given rows and columns
   DPtable(size_t rows, size_t cols, double defaultZ = 0.0)
      : table(rows, std::vector<Cell>(cols, Cell{defaultZ, {}})) {}


   tuple<double, int, vector<int>> queryMinCost(int maxNbrk, int t) const;
   void updateCell(int,int,double,vector<int>); // updates the cost of reaching time t with the given n segments
   bool isEmpty(int);
};