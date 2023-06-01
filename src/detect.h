//----------------------------------------------------------
// Copyright 2019 University of Oxford
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#ifndef DETECT_H
#define DETECT_H

#include <utility>
#include <string>
#include <vector>
#include "data_IO.h"
#include "alignment.h"
#include "tensor.h"

struct HMMdetection{

	public:
		std::map<unsigned int, double> refposToLikelihood;
		std::string readLikelihoodStdout;
};


/*function prototypes */
int detect_main( int argc, char** argv );

std::vector< unsigned int > getPOIs( std::string &, int );
double sequenceProbability( std::vector <double> &, std::string &, size_t, bool, PoreParameters, size_t, size_t );
std::map<unsigned int, std::pair<double,double>> runCNN_training(std::shared_ptr<AlignedRead> r, std::shared_ptr<ModelSession> session);
HMMdetection llAcrossRead( read &, unsigned int );

#endif
