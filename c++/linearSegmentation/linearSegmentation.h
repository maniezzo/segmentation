#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <ilcplex/cplex.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <tuple>
#include <numeric>   // accumulate
#include <algorithm> // for_each
#include <iomanip>   // setprecision
#include <utility>   // for std::pair
#include <cmath>     // for log function

using namespace std;

string baseDir;
string dsName;    // dataset name (file col .csv)
int maxIter;      // max nunm of lagrangian iterations
int maxTime;      // max secs of lagrangian runs
vector<vector<int>> rowids, colids;  // compression of indices, by row and by col

void writeListOLS(vector<tuple<int, int, double, double, double>> lstOLS, string dsName);
void postProcess(vector<tuple<int, int, double, double, double>>& lstOLS, vector<double>& x,int minlag);
int getSegmentId(int x0, int x1, vector<tuple<int, int, double, double, double>> lstOLS);
int get_line_intersection(double p0_x, double p0_y, double p1_x, double p1_y,
double p2_x, double p2_y, double p3_x, double p3_y, double* i_x, double* i_y);
void compressTableau(vector<tuple<int, int, double, double, double>> lstOLS);
int readConfig();
vector<int>  DAG_SSSP(int tinit, int tend, int, vector<tuple<int, int, double, double, double>> lstOLS);
vector<int> reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, vector<int> minsegm, int tinit, int tend);
void sortBasedOnAnother(std::vector<int>& v1, std::vector<int>& v2);
bool checkFeas(vector<int> sol, vector<tuple<int, int, double, double, double>> lstOLS, double expCost);
double calculateRSS(const std::vector<double>& y, const std::vector<double>& y_pred);
tuple<int, int, double, double, double> costAIC(int low, int up, vector<double> y);
tuple<int, int, double, double, double> costBIC(int low, int up, vector<double> y);
