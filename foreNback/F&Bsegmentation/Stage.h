#pragma once
#include "global.h"
#include <utility>   // needed to rename the results of the queries

// A stage, associated to a number of points from the start/end of the series
class Stage {
private:
   // key: num, value: {cost, {prefix_min_cost, prefix_min_num}}
   std::map<int, std::pair<double, std::pair<double,int>>> data;

public:
   void insert(int num, double cost);
   std::pair<int,double> queryMinCost(double threshold) const;
   std::pair<int,double> popMinCost(double threshold);
   void mainNode();
};
