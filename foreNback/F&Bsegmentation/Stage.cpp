#include "global.h"
#include "Stage.h"

/*
Use std::map<int, int> where key = num segm, value = minimum cost for that exact num of segments.
augment the structure to store prefix minimums.
On insertion, update prefix minimums.
On query, use upper_bound(threshold) to find the largest num <= threshold and return its prefix minimum.
*/

void Stage::insert(int num, int cost) 
{  auto it = data.lower_bound(num);

   if (it != data.end() && it->first == num) 
   {  // If same num exists, keep the smaller cost
      if (cost < it->second.first) 
         it->second.first = cost;
      else 
         return; // No update needed
   } 
   else // Insert new entry with temporary prefix min
      it = data.insert(it, {num, {cost, cost}});

   // Update prefix minimums from this point onwards
   int prevMin = (it==data.begin()) ? (std::numeric_limits<int>::max)()
      : std::prev(it)->second.second;
   it->second.second = (std::min)(prevMin, it->second.first);

   // Propagate updates forward if needed
   auto nextIt = std::next(it);
   while (nextIt!=data.end()&&nextIt->second.second>(std::min)(it->second.second, nextIt->second.first)) 
   {  nextIt->second.second = (std::min)(it->second.second, nextIt->second.first);
      ++nextIt;
   }
}

int  Stage::queryMinCost(int threshold) 
   const {
      auto it = data.upper_bound(threshold);
      if (it == data.begin()) 
         return -1; // No num <= threshold

      --it; // Largest num <= threshold
      return it->second.second;
   }

void Stage::mainNode() 
{
   Stage store;
   store.insert(10, 50);
   store.insert(5, 70);
   store.insert(3, 40);
   store.insert(8, 60);
   store.insert(5, 30); // Updates cost for num=5
   store.insert(5, 20); // Updates cost for num=5
   store.insert(5, 10); // Updates cost for num=5

   std::cout << "Min cost with num <= 4: " << store.queryMinCost(4) << "\n";  // 40
   std::cout << "Min cost with num <= 5: " << store.queryMinCost(5) << "\n";  // 30
   std::cout << "Min cost with num <= 9: " << store.queryMinCost(9) << "\n";  // 30
   std::cout << "Min cost with num <= 10: " << store.queryMinCost(10) << "\n"; // 30
   std::cout << "Min cost with num <= 2: " << store.queryMinCost(2) << "\n";  // -1
}