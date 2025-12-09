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
#include <queue>     // for min heap at each node
#include <functional>// for std::greater, sorting of the heaps (min)
#include <windows.h> // GetModuleFileName, for ExePath
#include <numeric>   // accumulate
#include <algorithm> // for_each
#include <iomanip>   // setprecision

using namespace std;

const double EPS = 0.00001;

string  baseDir;
string  dsName;       // dataset name (file col .csv)
int     maxNumEdges;  // max number if segments
clock_t tstart, tend;
double  ttot;
int n;                // num of points of the data series
int delta;            // width of the beam
int minLength;        // minmal segnment length
vector<double> Y;     // the dataseries to model
double zub;
int nFathomed;        // eliminated by the bound

struct Edge { int end1, end2, segm; double cost; }; // an edge in bellman ford
struct Node     // a node of the search tree
{  int t1, t2;  // initial, final time point
   double cost;  // segment cumulative cost   (current included)
   int nSegm;   // number of segments so far (current included)

   // For min-heap based on cost
   bool operator>(const Node& pNode) const 
   {  return cost>pNode.cost;
   }
};
vector<priority_queue<Node, vector<Node>, greater<Node>>> FminHeaps, BminHeaps;  // unexpanded nodes

void readConfig();
vector<string> split(string str, char sep);
int readData(string dataFileName, vector<int>& X, vector<double>& Y);
tuple<double, double> linearRegression(vector<int> x, vector<double> y);
tuple<int, int, double, double, double> costQRMSE(int t1, int t2);
void readSegments(string segmentFileName, vector<tuple<int, int, double, double, double>>& lstOLS);
void DAG_SSSP(vector<tuple<int, int, double, double, double>> lstOLS);
vector<int> reconstructSolution(vector<tuple<int, int, double, double, double>> lstOLS, vector<int> minsegm, int);
string exePath();
int run_BF(vector<tuple<int, int, double, double, double>> lstOLS, int maxNumEdges);
void reconstructBF(vector<double> costs, vector<Edge>& edges, int numv, int maxNumEdges);
void forward();
void generateFoffspring(int t1, int t2, int nSegm, double cost);  // forward offspring generation
void backward();
void generateBoffspring(int t1, int t2, int nSegm, double cost);  // backward offspring generation

