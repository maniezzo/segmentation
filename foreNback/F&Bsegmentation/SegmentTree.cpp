#include <bits/stdc++.h>
using namespace std;

struct SegTree {
   int n;
   vector<int> st;
   const int INF = 1e9;

   SegTree(int n): n(n), st(4*n, INF) {}

   void update(int p, int v, int idx, int l, int r) {
      if (l == r) { st[idx] = min(st[idx], v); return; }
      int m = (l + r) / 2;
      if (p <= m) update(p, v, idx*2, l, m);
      else        update(p, v, idx*2+1, m+1, r);
      st[idx] = min(st[idx*2], st[idx*2+1]);
   }

   void update(int p, int v) { update(p, v, 1, 0, n-1); }

   int query(int ql, int qr, int idx, int l, int r) {
      if (qr < l || r < ql) return INF;
      if (ql <= l && r <= qr) return st[idx];
      int m = (l+r)/2;
      return min(query(ql, qr, idx*2, l, m),
         query(ql, qr, idx*2+1, m+1, r));
   }

   int query(int p) { return query(0, p, 1, 0, n-1); } // min cost for num <= p
};

int main(){
   vector<int> nums = {1,3,5,10,100}; // example domain
   sort(nums.begin(), nums.end());
   nums.erase(unique(nums.begin(), nums.end()), nums.end());

   // compress num -> index
   auto idx = [&](int x) {
      return lower_bound(nums.begin(), nums.end(), x) - nums.begin();
      };

   SegTree seg(nums.size());

   seg.update(idx(5), 20);
   seg.update(idx(1), 50);
   seg.update(idx(3), 10);

   cout<<seg.query(idx(4))<<endl; // min cost for num <= 4

   return 0;
}