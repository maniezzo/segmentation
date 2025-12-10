#pragma once
#include "global.h"

// A stage, associated to a number of points from the start/end of the series
class Stage 
{
private:
   // key: num, value: {cost, prefix_min_cost}
   std::map<int, std::pair<int, int>> data;

public:
   void insert(int num, int cost);
   int  queryMinCost(int threshold) const;
   void mainNode();
};

