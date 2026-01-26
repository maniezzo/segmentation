#include "DPtable.h"

// updates the cost of reaching time t with the given n changepoints
void DPtable::updateCell(int nbrk,int t,double z,vector<int> chpt)
{  table[nbrk][t].z    = z;
   table[nbrk][t].chpt = chpt;
}

// minimo costo al tempo t usando al massimo maxBbrk changepoints
tuple<double, int, vector<int>> DPtable::queryMinCost(int maxNbrk, int t) const
{  double bestZ = std::numeric_limits<double>::infinity();
   int bestNbrk = -1;
   vector<int> bestChpt;

   for (int nbrk = 0; nbrk <= maxNbrk; ++nbrk)
   {
      const Cell& c = table[nbrk][t];

      if (c.z < bestZ)
      {
         bestZ    = c.z;
         bestNbrk = nbrk;
         bestChpt = c.chpt;
      }
   }

   return {bestZ, bestNbrk, bestChpt};
}

// check if a stage (time) has no open states
bool DPtable::isEmpty(int t)
{  bool empty = true;
   for (int nbrk = 0; nbrk <= maxNumEdges; ++nbrk)
      if(table[nbrk][t].z < DBL_MAX)
      {  empty = false;
         break;
      }

   return empty;
}