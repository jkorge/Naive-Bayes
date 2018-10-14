#ifndef CLASSIFIER_H
#define CLASSIFIER_H
#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <algorithm>

using namespace std;

class GNB {
public:

	//vector<string> possible_labels = {"left","keep","right"};

	vector<string> possible_labels;
	map<string, vector<double> > means;
	map<string, vector<double> > stddevs;
	map<string, double> counts;
	double total_count;


	/**
  	* Constructor
  	*/
 	GNB();

	/**
 	* Destructor
 	*/
 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  string predict(vector<double>);

  void printMap(map<string, vector<double> > mp);

};

#endif