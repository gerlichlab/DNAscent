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

/*function prototypes */
int sense_main( int argc, char** argv );

struct DetectedRead{

	std::vector< unsigned int > positions;
	std::vector< double > brduCalls;
	std::string readID, chromosome, strand;
	int mappingLower, mappingUpper;
};

#endif
