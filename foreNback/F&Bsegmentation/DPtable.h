#pragma once
#include "global.h"

// table of the dynamic programming recursion, it holds stage dat
class DPtable {
private:

public:
   struct Cell {bool isExpanded; double z; vector<int> chpt;}; // i changepoints che hanno portato al costo z
   vector<vector<Cell>> table;
   // Constructor: initialize the table with given rows and columns
   DPtable(size_t rows, size_t cols, double defaultZ = DBL_MAX)
      : table(rows, std::vector<Cell>(cols, Cell{false, defaultZ, {}})) {}


   tuple<double, int, vector<int>> queryMinCost(int maxNbrk, int t) const;
   void updateCell(bool, int, int, double,vector<int>); // updates the cost of reaching time t with the given n segments
   bool isEmpty(int);
};