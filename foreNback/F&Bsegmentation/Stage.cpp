#include "global.h"
#include "Stage.h"

/*
Use std::map<int, int> where key = num segm, value = minimum cost for that exact num of segments.
augment the structure to store prefix minimums.
On insertion, update prefix minimums.
On query, use upper_bound(threshold) to find the largest num <= threshold and return its prefix minimum.
*/

// Insert a new (cost,num,vector) tuple
void Stage::insert(int num, double cost, const std::vector<int>& vec)
{
   auto it = data.lower_bound(num);

   if (it != data.end() && it->first == num) 
   {
      double& existingCost = std::get<0>(it->second);
      if (cost < existingCost) 
      {  existingCost = cost;
         std::get<1>(it->second) = vec;
      } 
      else return;
   } 
   else 
      it = data.insert(it, std::make_pair(num, Value(cost, vec, PrefixMin(cost, num))));

   double prevMinCost = (it == data.begin())
      ? std::numeric_limits<double>::infinity()
      : std::get<2>(std::prev(it)->second).first;

   int prevMinNum = (it == data.begin())
      ? -1
      : std::get<2>(std::prev(it)->second).second;

   double currCost = std::get<0>(it->second);
   if (currCost < prevMinCost) 
      std::get<2>(it->second) = PrefixMin(currCost, it->first);
   else 
      std::get<2>(it->second) = PrefixMin(prevMinCost, prevMinNum);

   auto nextIt = std::next(it);
   while (nextIt != data.end()) 
   {
      double nextCost = std::get<0>(nextIt->second);
      const PrefixMin& leftMin = std::get<2>(it->second);
      PrefixMin& nextMin = std::get<2>(nextIt->second);

      double combinedMinCost = (std::min)(leftMin.first, nextCost);
      if (nextMin.first <= combinedMinCost) break;

      if (nextCost < leftMin.first) 
         nextMin = PrefixMin(nextCost, nextIt->first);
      else
         nextMin = leftMin;

      ++nextIt;
   }
}

// Query the minimum cost and its num for all entries <= threshold. Returns {minCost, minNum}
std::tuple<double, int, std::vector<int>> Stage::queryMinCost(double threshold) const
{
   auto it = data.upper_bound(static_cast<int>(threshold));
   if (it == data.begin()) return std::make_tuple(0.0, -1, std::vector<int>());

   --it;
   const PrefixMin& pm = std::get<2>(it->second);
   int minNum = pm.second;
   double minCost = pm.first;

   auto targetIt = data.find(minNum);
   if (targetIt == data.end()) return std::make_tuple(0.0, -1, std::vector<int>());

   const std::vector<int>& vec = std::get<1>(targetIt->second);
   return std::make_tuple(minCost, minNum, vec);
}

// pop the least cost element, retrieves its data and erases it
std::tuple<double, int, std::vector<int>> Stage::popMinCost(double threshold)
{
   auto it = data.upper_bound(static_cast<int>(threshold));
   if (it == data.begin()) return std::make_tuple(0.0, -1, std::vector<int>());

   --it;
   const PrefixMin& pm = std::get<2>(it->second);
   int minNum = pm.second;
   double minCost = pm.first;

   auto targetIt = data.find(minNum);
   if (targetIt == data.end()) return std::make_tuple(0.0, -1, std::vector<int>());

   std::vector<int> vec = std::get<1>(targetIt->second);
   auto result = std::make_tuple(std::get<0>(targetIt->second), minNum, vec);

   data.erase(targetIt);

   // Recompute prefix minima
   double runningMin = std::numeric_limits<double>::infinity();
   int runningNum = -1;
   for (auto &kv : data) 
   {
      double cost = std::get<0>(kv.second);
      if (cost < runningMin) 
      {  runningMin = cost;
         runningNum = kv.first;
      }
      std::get<2>(kv.second) = PrefixMin(runningMin, runningNum);
   }

   return result;
}

void Stage::mainNode()
{  Stage s;

   s.insert(5, 10.0, {1,2,3});
   s.insert(8, 7.0,  {4,5});
   s.insert(12, 15.0,{6});
   s.insert(3, 20.0, {7,8,9});
   s.insert(10, 5.0, {10});
   s.insert(2,  2.0, {7,8,9,10});

   double minCost;
   int minNum;
   std::vector<int> vec;

   std::tie(minCost, minNum, vec) = s.queryMinCost(12);

   std::cout << "Threshold 12 -> minNum=" << minNum
      << ", minCost=" << minCost
      << ", vec size=" << vec.size() << "\n";
}