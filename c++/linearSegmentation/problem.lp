\ Model ExtendedSetCover
\ LP format - for model browsing. Use MPS format to capture full model detail.
\ Signature: 0xf135aab609d82c56
Minimize
  10 x0 + 8 x1 + 15 x2
Subject To
 c0: x0 + x2 >= 1
 c1: x0 + x1 + x2 >= 1
 c2: x0 + x1 + x2 >= 1
 c3: x1 + x2 >= 1
 c4: x1 + x2 >= 1
Bounds
Binaries
 x0 x1 x2
End
