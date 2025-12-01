#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <tuple>
#include <stack>
#include <windows.h> // GetModuleFileName, for ExePath
#include <numeric>   // accumulate
#include <algorithm> // for_each
#include <iomanip>   // setprecision

using namespace std;

const double EPS = 0.00001;

string baseDir;
string dsName;    // dataset name (file col .csv)
int maxNumEdges;  // max number if segments
clock_t tstart, tend;
double ttot;

struct Edge { int end1, end2, segm; double cost; }; // an edge in bellman ford

void readConfig();
vector<string> split(string str, char sep);
int readData(string dataFileName, vector<int>& X, vector<double>& Y);
tuple<double, double> linearRegression(vector<int> x, vector<double> y);
tuple<int, int, double, double, double> costQRMSE(int low, int up, vector<double> y);
void readSegments(string segmentFileName, vector<tuple<int, int, double, double, double>>& lstOLS);
void DAG_SSSP(vector<tuple<int, int, double, double, double>> lstOLS);
vector<int> reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, vector<int> minsegm, int);
string exePath();
int run_BF(vector<tuple<int, int, double, double, double>> lstOLS, int maxNumEdges);
void reconstructBF(vector<double> costs, vector<Edge>& edges, int numv, int maxNumEdges);
