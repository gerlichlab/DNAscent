//----------------------------------------------------------
// Copyright 2017 University of Oxford
// Written by Michael A. Boemo (michael.boemo@path.ox.ac.uk)
// This software is licensed under GPL-2.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#include "../tensorflow/include/tensorflow/c/c_api.h"
#include <fstream>
#include "regions.h"
#include "data_IO.h"
#include "error_handling.h"
#include "train.h"
#include "sense.h"
#include <cmath>
#include <math.h>
#include <algorithm>

static const char *help=
"forkSense: DNAscent executable that calls replication origins, fork movement, and fork stalling.\n"
"To run DNAscent forkSense, do:\n"
"  ./DNAscent forkSense [arguments]\n"
"Example:\n"
"  ./DNAscent forkSense -d /path/to/BrdUCalls.detect -o /path/to/output.forkSense\n"
"Required arguments are:\n"
"  -d,--detect               path to output file from DNAscent detect,\n"
"  -o,--output               path to output file for forkSense.\n"
"Optional arguments are:\n";

 struct Arguments {

	std::string detectFilename;
	std::string outputFilename;
};

Arguments parseSenseArguments( int argc, char** argv ){

 	if( argc < 2 ){
 		std::cout << "Exiting with error.  Insufficient arguments passed to DNAscent regions." << std::endl << help << std::endl;
		exit(EXIT_FAILURE);
	}
 	if ( std::string( argv[ 1 ] ) == "-h" or std::string( argv[ 1 ] ) == "--help" ){
 		std::cout << help << std::endl;
		exit(EXIT_SUCCESS);
	}
	else if( argc < 4 ){
 		std::cout << "Exiting with error.  Insufficient arguments passed to DNAscent regions." << std::endl;
		exit(EXIT_FAILURE);
	}

 	Arguments args;

 	/*parse the command line arguments */
	for ( int i = 1; i < argc; ){

 		std::string flag( argv[ i ] );

 		if ( flag == "-d" or flag == "--detect" ){
 			std::string strArg( argv[ i + 1 ] );
			args.detectFilename = strArg;
			i+=2;
		}
		else if ( flag == "-o" or flag == "--output" ){
 			std::string strArg( argv[ i + 1 ] );
			args.outputFilename = strArg;
			i+=2;
		}
		else throw InvalidOption( flag );
	}
	return args;
}


int sense_main( int argc, char** argv ){

	Arguments args = parseSenseArguments( argc, argv );

	//get a read count
	int readCount = 0;
	std::string line;
	std::ifstream inFile( args.detectFilename );
	if ( not inFile.is_open() ) throw IOerror( args.detectFilename );
	while( std::getline( inFile, line ) ){

		if ( line.substr(0,1) == ">" ) readCount++;
	}	
	progressBar pb(readCount,false);
	inFile.close();

 	inFile.open( args.detectFilename );
	if ( not inFile.is_open() ) throw IOerror( args.detectFilename );
 	std::ofstream outFile( args.outputFilename );
	if ( not outFile.is_open() ) throw IOerror( args.outputFilename );

	std::vector< DetectedRead > buffer;
	int progress = 0;
	while( std::getline( inFile, line ) ){

		if ( line.substr(0,1) == ">" ){

			progress++;
			pb.displayProgress( progress, 0, 0 );

			DetectedRead d;

			std::stringstream ssLine(line);
			std::string column;
			int cIndex = 0;
			while ( std::getline( ssLine, column, ' ' ) ){

				if ( cIndex == 0 ) d.readID = column;
				else if ( cIndex == 1 ) d.chromosome = column;
				else if ( cIndex == 2 ) d.mappingLower = std::stoi(column);
				else if ( cIndex == 3 ) d.mappingUpper = std::stoi(column);
				else if ( cIndex == 4 ) d.strand = column;
				else throw DetectParsing();
				cIndex++;
			}
			assert(d.mappingUpper > d.mappingLower);
			buffer.push_back(d);
		}
		else{

			std::string column;
			std::stringstream ssLine(line);
			int position = -1, cIndex = 0;
			AnalogueScore B, BM;
			int countCol = std::count(line.begin(), line.end(), '\t');
			while ( std::getline( ssLine, column, '\t' ) ){

				if ( cIndex == 0 ){

					position = std::stoi(column);
				}
				else if ( cIndex == 1 ){

					B.set(std::stof(column));
				}
				else if ( cIndex == 2 and countCol > 3 ){ //methyl-aware detect file

					BM.set(std::stof(column));
				}
				cIndex++;
			}
			buffer.back().positions.push_back(position);
			buffer.back().brduCalls.push_back(B.get());
		}
	}

	//empty the buffer at the end


	inFile.close();
	outFile.close();
	std::cout << std::endl << "Done." << std::endl;

	return 0;
}
