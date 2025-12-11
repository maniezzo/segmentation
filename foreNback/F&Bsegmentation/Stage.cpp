#include "global.h"
#include "Stage.h"

/*
Use std::map<int, int> where key = num segm, value = minimum cost for that exact num of segments.
augment the structure to store prefix minimums.
On insertion, update prefix minimums.
On query, use upper_bound(threshold) to find the largest num <= threshold and return its prefix minimum.
*/

// Insert a new (cost,num) pair
void Stage::insert(int num, double cost) {
   auto it = data.lower_bound(num);

   if (it != data.end() && it->first == num) {
      if (cost < it->second.first)
         it->second.first = cost;
      else
         return;
   } else {
      it = data.insert(it, {num, {cost, {cost, num}}});
   }

   double prevMinCost = (it==data.begin()) ? std::numeric_limits<double>::infinity()
      : std::prev(it)->second.second.first;
   int prevMinNum = (it==data.begin()) ? -1
      : std::prev(it)->second.second.second;

   if (it->second.first < prevMinCost) {
      it->second.second = {it->second.first, num};
   } else {
      it->second.second = {prevMinCost, prevMinNum};
   }

   auto nextIt = std::next(it);
   while (nextIt != data.end() &&
      nextIt->second.second.first>(std::min)(it->second.second.first, nextIt->second.first)) {
      if (nextIt->second.first < it->second.second.first)
         nextIt->second.second = {nextIt->second.first, nextIt->first};
      else
         nextIt->second.second = it->second.second;
      ++nextIt;
   }
}

// Query the minimum cost and its num for all entries <= threshold
std::pair<double,int> Stage::queryMinCost(double threshold) const {
   auto it = data.upper_bound(threshold);
   if (it == data.begin())
      return {-1, -1}; // No num <= threshold

   --it;
   return it->second.second; // returns {minCost, minNum}
}

// pop the least cost element, retrieves its data and erases it
std::pair<double,int> Stage::popMinCost(double threshold) 
{
   auto it = data.upper_bound(threshold);
   if (it == data.begin())
      return {-1, -1}; // No num <= threshold

   --it;
   // This is the prefix minimum info at this point
   int minNum   = it->second.second.second;
   double minCost = it->second.second.first;

   // Find the actual entry with key = minNum
   auto targetIt = data.find(minNum);
   if (targetIt == data.end())
      return {-1, -1}; // Shouldn't happen

   // Save result
   std::pair<double,int> result{targetIt->second.first,minNum};

   // Erase the element
   data.erase(targetIt);

   // Recompute prefix minima from scratch
   double runningMin = std::numeric_limits<double>::infinity();
   int runningNum = -1;
   for (auto &kv : data) {
      if (kv.second.first < runningMin) {
         runningMin = kv.second.first;
         runningNum = kv.first;
      }
      kv.second.second = {runningMin, runningNum};
   }

   return result;
}

void Stage::mainNode() 
{  Stage s;

   // Insert some test data
   s.insert(5, 10);
   s.insert(8, 7);
   s.insert(12, 15);
   s.insert(3, 20);
   s.insert(10, 5);

   // Query with different thresholds
   auto varb = s.queryMinCost(12);
   std::cout << "Threshold 12 -> minNum=" << varb.first << ", minCost=" << varb.second << "\n";

   return;
}