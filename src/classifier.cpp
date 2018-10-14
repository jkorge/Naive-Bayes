#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include <map>
#include <algorithm>
#include "classifier.h"

/**
 * Initializes GNB
 */
GNB::GNB() {}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{

	/*
		Trains the classifier with N data points and labels.

		INPUTS
		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/

	//Remove duplicates from training labels and save as possible_labels
	possible_labels = labels;
	sort(possible_labels.begin(), possible_labels.end());
	possible_labels.erase(unique(possible_labels.begin(), possible_labels.end()), possible_labels.end());	

	//Replace d with d mod 4
	for(unsigned int i=0;i<data.size();i++){
		data[i][1] = fmod(data[i][1],4.0);
	}

	//Initiate means, stddevs, and counts with zeros
  for(unsigned int i=0;i<labels.size();i++){
      means[labels[i]] = vector<double> (data[i].size(), 0);
      stddevs[labels[i]] = vector<double> (data[i].size(), 0);
      counts[labels[i]] = 0;
  }

  /*
  cout << "Initial Means:" << endl;
  printMap(means);
	cout << "Initial Standard Deviations:" << endl;
  printMap(stddevs);
  */

  //Count number of training-data vectors
  total_count = labels.size();

  //Compute sums for means
	for(unsigned int i=0;i<labels.size();i++){
		counts[labels[i]] += 1;
		for(unsigned int j=0;j<data[i].size();j++){
			means[labels[i]][j] += data[i][j];
		}
	}

	//Divide by counts to get means
	for(unsigned int i=0;i<possible_labels.size();i++){
		for(unsigned int j=0;j<means[possible_labels[i]].size();j++){
			means[possible_labels[i]][j] /= counts[possible_labels[i]];
		}
	}

	//Accumulate squared difference from means for stddevs
	for(unsigned int i=0;i<data.size();i++){
		for(unsigned int j=0;j<data[i].size();j++){
			stddevs[labels[i]][j] += pow(data[i][j]-means[labels[i]][j], 2);
		}
	}

	//Take sqrt to get stddevs
	for(unsigned int i=0;i<possible_labels.size();i++){
		for(unsigned int j=0;j<stddevs[possible_labels[i]].size();j++){
			stddevs[possible_labels[i]][j] = sqrt(stddevs[possible_labels[i]][j] / counts[labels[i]]);
		}
	}

	/*
	cout << "Final Means:" << endl;
  printMap(means);
	cout << "Final Standard Deviations:" << endl;
  printMap(stddevs);
  */

	/*
	cout << "Number of each label" << endl;
  for(const auto &p : counts){
  	cout << p.first << ": " << p.second << endl;
  }
  */

}

string GNB::predict(vector<double> sample)
{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""
	*/

	map<string, vector<double> > probabilites;
	vector<double> label_prob(sample.size(), 0);
	double max_prob = 0.0;
	int max_index;

	//For each label, produce a vector of probabilites
	for(unsigned int i=0;i<possible_labels.size();i++){
		for(unsigned int j=0;j<sample.size();j++){

			//Probability of x given some normally distributed label:
			//	(1/sqrt(2 * pi * sigma^2)) * exp( (-(x - mu)^2) / (2 * sigma^2) )

			double expt_num = -0.5*pow(sample[j]-means[possible_labels[i]][j], 2);
			double expt_denom = pow(stddevs[possible_labels[i]][j], 2);
			double expt = expt_num/expt_denom;
			double coeff = 1.0/sqrt(2.0*M_PI*pow(stddevs[possible_labels[i]][j], 2));
			
			label_prob[j] = coeff*exp(expt);
			
			
		}

		probabilites[possible_labels[i]] = label_prob;
	}

	//For each label, take the product over the probability vector
	for(unsigned int i=0;i<possible_labels.size();i++){
		double prob = 1.0;

		for(unsigned int j=0;j<probabilites[possible_labels[i]].size();j++){
			prob *= probabilites[possible_labels[i]][j];
		}
		

		//Multiply by the a-priori probability of the label
		prob *= (counts[possible_labels[i]]/total_count);
		//cout << "Label: " << possible_labels[i] << endl << "Probability: " << prob << endl << endl;
		//Track most likely outcome
		if(prob > max_prob){
			max_prob = prob;
			max_index = i;
		}
	}

	return possible_labels[max_index];

}

void GNB::printMap(map<string, vector<double> > mp){
  for(const auto &p : mp){
  	cout << p.first << " ";
  	for(unsigned int i=0;i<p.second.size();i++){
  		cout << p.second[i] << " ";
  	}
  	cout << endl;
  }
}