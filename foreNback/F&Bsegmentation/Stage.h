#pragma once
#include "global.h"
#include <utility>   // needed to rename the results of the queries

// A stage, associated to a number of points from the start/end of the series
class Stage {
private:
   // structure: {cost, vec, {prefix_min_cost, prefix_min_num}}. vec is a list of changepoints
   typedef std::pair<double, int> PrefixMin;
   typedef std::tuple<double, std::vector<int>, PrefixMin> Value;
   std::map<int, Value> data; // key = num

public:
   bool isEmpty() const { return data.empty(); }
   void insert(int num, double cost, const std::vector<int>& vec);
   tuple<double, int, vector<int>> queryMinCost(double threshold) const;
   tuple<double, int, vector<int>> popMinCost(double threshold);
   void mainNode();
};