//----------------------------------------------------------
// Copyright 2017 University of Oxford
// Written by Michael A. Boemo (michael.boemo@path.ox.ac.uk)
// This software is licensed under GPL-2.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#ifndef SENSE_H
#define SENSE_H

#include <cassert>
#include <vector>

/*function prototypes */
int sense_main( int argc, char** argv );

class DetectedRead{

	public:
		std::vector< unsigned int > positions;
		std::vector< double > brduCalls;
		std::string readID, chromosome, strand, header;
		int mappingLower, mappingUpper;
		std::vector<std::vector<float>> probabilities;
		std::vector<std::pair<int,int>> stalls;
		std::vector<std::pair<int,int>> origins;
		void trim(unsigned int trimFactor){

			assert(positions.size() > trimFactor and brduCalls.size() > trimFactor and positions.size() == brduCalls.size());
			unsigned int cropFromEnd = positions.size() % trimFactor;
			brduCalls.erase(brduCalls.end() - cropFromEnd, brduCalls.end());
			positions.erase(positions.end() - cropFromEnd, positions.end());
			assert(positions.size() % trimFactor == 0 and brduCalls.size() % trimFactor == 0);
		}
};

#endif
