//----------------------------------------------------------
// Copyright 2020 University of Cambridge
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#include <fstream>
#include "detect.h"
#include <math.h>
#include <utility>
#include <stdlib.h>
#include <limits>
#include "common.h"
#include "event_handling.h"
#include "../fast5/include/fast5.hpp"
#include "poreModels.h"
#include "alignment.h"
#include "error_handling.h"

#include "../Penthus/src/hmm.h"
#include "../Penthus/src/probability.h"
#include "../Penthus/src/states.h"


static const char *help=
"align: DNAscent executable that generates a BrdU-aware event alignment.\n"
"To run DNAscent align, do:\n"
		"  ./DNAscent align -b /path/to/alignment.bam -r /path/to/reference.fasta -i /path/to/index.dnascent -o /path/to/output.detect\n"
"Required arguments are:\n"
"  -b,--bam                  path to alignment BAM file,\n"
"  -r,--reference            path to genome reference in fasta format,\n"
"  -i,--index                path to DNAscent index,\n"
"  -o,--output               path to output file that will be generated.\n"
"Optional arguments are:\n"
"  -t,--threads              number of threads (default is 1 thread),\n"
"  --methyl-aware            account for CpG, Dcm, and Dam methylation in BrdU calling,\n"
"  -m,--maxReads             maximum number of reads to consider,\n"
"  -q,--quality              minimum mapping quality (default is 20),\n"
"  -l,--length               minimum read length in bp (default is 100).\n"
"Written by Michael Boemo, Department of Pathology, University of Cambridge.\n"
"Please submit bug reports to GitHub Issues (https://github.com/MBoemo/DNAscent/issues).";

struct Arguments {
	std::string bamFilename;
	std::string referenceFilename;
	std::string outputFilename;
	std::string indexFilename;
	bool methylAware, capReads;
	double divergence;
	int minQ, maxReads;
	int minL;
	unsigned int threads;
};

Arguments parseAlignArguments( int argc, char** argv ){

	if( argc < 2 ){

		std::cout << "Exiting with error.  Insufficient arguments passed to DNAscent detect." << std::endl << help << std::endl;
		exit(EXIT_FAILURE);
	}

	if ( std::string( argv[ 1 ] ) == "-h" or std::string( argv[ 1 ] ) == "--help" ){

		std::cout << help << std::endl;
		exit(EXIT_SUCCESS);
	}
	else if( argc < 4 ){

		std::cout << "Exiting with error.  Insufficient arguments passed to DNAscent detect." << std::endl;
		exit(EXIT_FAILURE);
	}

	Arguments args;

	/*defaults - we'll override these if the option was specified by the user */
	args.threads = 1;
	args.minQ = 20;
	args.minL = 100;
	args.methylAware = false;
	args.divergence = 0;
	args.capReads = false;
	args.maxReads = 0;

	/*parse the command line arguments */

	for ( int i = 1; i < argc; ){

		std::string flag( argv[ i ] );

		if ( flag == "-b" or flag == "--bam" ){

			std::string strArg( argv[ i + 1 ] );
			args.bamFilename = strArg;
			i+=2;
		}
		else if ( flag == "-r" or flag == "--reference" ){

			std::string strArg( argv[ i + 1 ] );
			args.referenceFilename = strArg;
			i+=2;
		}
		else if ( flag == "-t" or flag == "--threads" ){

			std::string strArg( argv[ i + 1 ] );
			args.threads = std::stoi( strArg.c_str() );
			i+=2;
		}
		else if ( flag == "-q" or flag == "--quality" ){

			std::string strArg( argv[ i + 1 ] );
			args.minQ = std::stoi( strArg.c_str() );
			i+=2;
		}
		else if ( flag == "-l" or flag == "--length" ){

			std::string strArg( argv[ i + 1 ] );
			args.minL = std::stoi( strArg.c_str() );
			i+=2;
		}
		else if ( flag == "-i" or flag == "--index" ){

			std::string strArg( argv[ i + 1 ] );
			args.indexFilename = strArg;
			i+=2;
		}
		else if ( flag == "-o" or flag == "--output" ){

			std::string strArg( argv[ i + 1 ] );
			args.outputFilename = strArg;
			i+=2;
		}
		else if ( flag == "-m" or flag == "--maxReads" ){

			std::string strArg( argv[ i + 1 ] );
			args.capReads = true;
			args.maxReads = std::stoi( strArg.c_str() );
			i+=2;
		}
		else if ( flag == "--divergence" ){

			std::string strArg( argv[ i + 1 ] );
			args.divergence = std::stof(strArg.c_str());
			i+=2;
		}
		else if ( flag == "--methyl-aware" ){

			args.methylAware = true;
			i+=1;
		}
		else throw InvalidOption( flag );
	}
	if (args.outputFilename == args.indexFilename or args.outputFilename == args.referenceFilename or args.outputFilename == args.bamFilename) throw OverwriteFailure();

	return args;
}


std::pair< double, std::vector< std::string > > eventViterbi( std::vector <double> &observations,
														      std::string &sequence,
															  PoreParameters scalings){

	//Initial transitions within modules (internal transitions)
	double internalM12I = 0.001;
	double internalI2I = 0.001;
	double internalM12M1 = 1. - (1./scalings.eventsPerBase);//0.4;

	//Initial transitions between modules (external transitions)
	double externalD2D = 0.3;
	double externalD2M1 = 0.7;
	double externalI2M1 = 0.999;
	double externalM12D = 0.0025;
	double externalM12M1 = 1.0 - externalM12D - internalM12I - internalM12M1;//0.5965;

	HiddenMarkovModel hmm = HiddenMarkovModel();

	/*STATES - vector (of vectors) to hold the states at each position on the reference - fill with dummy values */
	std::vector< std::vector< State > > states( 3, std::vector< State >( sequence.length() - 5, State( NULL, "", "", "", 1.0 ) ) );

	/*DISTRIBUTIONS - vector to hold normal distributions, a single uniform and silent distribution to use for everything else */
	std::vector< NormalDistribution > nd;
	nd.reserve( sequence.length() - 5 );

	SilentDistribution sd( 0.0, 0.0 );
	UniformDistribution ud( 0, 250.0 );

	std::string loc, sixMer;

	/*create make normal distributions for each reference position using the ONT 6mer model */
	for ( unsigned int i = 0; i < sequence.length() - 5; i++ ){

		sixMer = sequence.substr( i, 6 );
		nd.push_back( NormalDistribution( scalings.shift + scalings.scale * thymidineModel.at(sixMer).first, scalings.var * thymidineModel.at(sixMer).second ) );
	}

	/*the first insertion state after start */
	//State firstI = State( &ud, "-1_I", "", "", 1.0 );
	//hmm.add_state( firstI );

	/*add states to the model, handle internal module transitions */
	for ( unsigned int i = 0; i < sequence.length() - 5; i++ ){

		loc = std::to_string( i );
		sixMer = sequence.substr( i, 6 );

		states[ 0 ][ i ] = State( &sd,		loc + "_D", 	sixMer,	"", 		1.0 );
		states[ 1 ][ i ] = State( &ud,		loc + "_I", 	sixMer,	"", 		1.0 );
		states[ 2 ][ i ] = State( &nd[i], 	loc + "_M", 	sixMer,	loc + "_match", 1.0 );

		/*add state to the model */
		for ( unsigned int j = 0; j < 3; j++ ){

			states[ j ][ i ].meta = sixMer;
			hmm.add_state( states[ j ][ i ] );
		}

		/*transitions between states, internal to a single base */
		/*from I */
		hmm.add_transition( states[1][i], states[1][i], internalI2I );

		/*from M1 */
		hmm.add_transition( states[2][i], states[2][i], internalM12M1 );
		hmm.add_transition( states[2][i], states[1][i], internalM12I );
	}

	/*add transitions between modules (external transitions) */
	for ( unsigned int i = 0; i < sequence.length() - 6; i++ ){

		/*from D */
		hmm.add_transition( states[0][i], states[0][i + 1], externalD2D );
		hmm.add_transition( states[0][i], states[2][i + 1], externalD2M1 );

		/*from I */
		hmm.add_transition( states[1][i], states[2][i + 1], externalI2M1 );

		/*from M */
		hmm.add_transition( states[2][i], states[0][i + 1], externalM12D );
		hmm.add_transition( states[2][i], states[2][i + 1], externalM12M1 );
	}

	/*handle start states */
	//hmm.add_transition( hmm.start, firstI, 0.25 );
	hmm.add_transition( hmm.start, states[0][0], externalM12D );
	hmm.add_transition( hmm.start, states[1][0], internalM12I );
	hmm.add_transition( hmm.start, states[2][0], externalM12M1 + internalM12M1);

	/*transitions from first insertion */
	//hmm.add_transition( firstI, firstI, 0.25 );
	//hmm.add_transition( firstI, states[0][0], 0.25 );
	//hmm.add_transition( firstI, states[2][0], 0.5 );

	/*handle end states */
	hmm.add_transition( states[0][sequence.length() - 6], hmm.end, 1.0 );
	hmm.add_transition( states[1][sequence.length() - 6], hmm.end, externalI2M1 );
	hmm.add_transition( states[2][sequence.length() - 6], hmm.end, externalM12M1 + externalM12D );

	hmm.finalise();
	return hmm.viterbi( observations );
}


std::string eventalign( read &r,
            unsigned int windowLength ){

	std::string out;
	//get the positions on the reference subsequence where we could attempt to make a call
	std::string strand;
	unsigned int readHead = 0;
	if ( r.isReverse ) strand = "rev";
	else strand = "fwd";

	out += ">" + r.readID + " " + r.referenceMappedTo + " " + std::to_string(r.refStart) + " " + std::to_string(r.refEnd) + " " + strand + "\n";

	unsigned int posOnRef = 0;
	while ( posOnRef < r.referenceSeqMappedTo.size() - windowLength - 7 ){ //-7 because we need to reach forward 6 bases when we process the match states

		std::string readSnippet = (r.referenceSeqMappedTo).substr(posOnRef, windowLength);

		//make sure the read snippet is fully defined as A/T/G/C in reference
		unsigned int As = 0, Ts = 0, Cs = 0, Gs = 0;
		for ( std::string::iterator i = readSnippet.begin(); i < readSnippet.end(); i++ ){

			switch( *i ){
				case 'A' :
					As++;
					break;
				case 'T' :
					Ts++;
					break;
				case 'G' :
					Gs++;
					break;
				case 'C' :
					Cs++;
					break;
			}
		}
		if ( readSnippet.length() != (As + Ts + Gs + Cs) ){
			continue;
		}
		std::vector< double > eventSnippet;
		std::vector< double > eventLengthsSnippet;

		/*get the events that correspond to the read snippet */
		for ( unsigned int j = readHead; j < (r.eventAlignment).size(); j++ ){

			/*if an event has been aligned to a position in the window, add it */
			if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef] and (r.eventAlignment)[j].second < (r.refToQuery)[posOnRef + windowLength] ){

				double ev = (r.normalisedEvents)[(r.eventAlignment)[j].first];
				if (ev > r.scalings.shift + 1.0 and ev < 250.0){
					eventSnippet.push_back( ev );
					eventLengthsSnippet.push_back( (r.eventLengths)[(r.eventAlignment)[j].first] );
				}
			}

			/*stop once we get to the end of the window */
			if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef + windowLength] ) break;
		}

		//pass on this window if we have a deletion
		//TODO: make sure this does actually catch deletion cases properly
		if ( eventSnippet.size() < 2 ){

			posOnRef += windowLength;
			continue;
		}

		/*
		TESTING - print out the read snippet, the ONT model, and the aligned events
		std::cout << readSnippet << std::endl;
		for ( int pos = 0; pos < readSnippet.length()-5; pos++ ){

			std::cout << readSnippet.substr(pos,6) << "\t" << thymidineModel.at( readSnippet.substr(pos,6) ).first << std::endl;
		}
		for ( auto ev = eventSnippet.begin(); ev < eventSnippet.end(); ev++){
			double scaledEv =  (*ev - r.scalings.shift) / r.scalings.scale;
			std::cout << scaledEv << std::endl;
		}
		*/

		//calculate where we are on the assembly - if we're a reverse complement, we're moving backwards down the reference genome
		int globalPosOnRef;
		if ( r.isReverse ) globalPosOnRef = r.refEnd - posOnRef - 6;
		else globalPosOnRef = r.refStart + posOnRef;

		std::pair< double, std::vector< std::string > > localAlignment = eventViterbi( eventSnippet, readSnippet, r.scalings);

		std::vector< std::string > stateLabels = localAlignment.second;
		size_t lastM_ev = 0;
		size_t lastM_ref = 0;

		size_t evIdx = 0;

		//grab the index of the last match so we don't print insertions where we shouldn't
		for (size_t i = 1; i < stateLabels.size(); i++){

			assert(stateLabels[i] != "START");
			if (stateLabels[i] == "END") continue;

			std::string label = stateLabels[i].substr(stateLabels[i].find('_')+1);

	        if (label == "D") continue; //silent states don't emit an event

	        int pos = std::stoi(stateLabels[i].substr(0,stateLabels[i].find('_')));

	        if (label == "M"){
	        	lastM_ev = evIdx;
	        	lastM_ref = pos;
	        }
	        evIdx ++;
		}

		//do a second pass to print the alignment
		evIdx = 0;
		for (size_t i = 1; i < stateLabels.size(); i++){

			assert(stateLabels[i] != "START");
			if (stateLabels[i] == "END") continue;

			std::string label = stateLabels[i].substr(stateLabels[i].find('_')+1);

	        if (label == "D") continue; //silent states don't emit an event

	        int pos = std::stoi(stateLabels[i].substr(0,stateLabels[i].find('_')));
			std::string sixMerStrand = (r.referenceSeqMappedTo).substr(posOnRef + pos, 6);

			double scaledEvent = (eventSnippet[evIdx] - r.scalings.shift) / r.scalings.scale;
			double eventLength = eventLengthsSnippet[evIdx];

			assert(scaledEvent > 0.0);

			unsigned int evPos;
			std::string sixMerRef;
			if (r.isReverse){
				evPos = globalPosOnRef - pos;
				sixMerRef = reverseComplement(sixMerStrand);
			}
			else{
				evPos = globalPosOnRef + pos;
				sixMerRef = sixMerStrand;
			}

			if (label == "M"){
				out += std::to_string(evPos) + "\t" + sixMerRef + "\t" + std::to_string(scaledEvent) + "\t" + std::to_string(eventLength) + "\t" + sixMerStrand + "\t" + std::to_string(thymidineModel.at(sixMerStrand).first) + "\t" + std::to_string(thymidineModel.at(sixMerStrand).second) + "\n";
			}
			else if (label == "I" and evIdx < lastM_ev){ //don't print insertions after the last match because we're going to align these in the next segment
				out += std::to_string(evPos) + "\t" + sixMerRef + "\t" + std::to_string(scaledEvent) + "\t" + std::to_string(eventLength) + "\t" + "NNNNNN" + "\t" + "0" + "\t" + "0" + "\n";
			}

	        evIdx ++;
		}

		//TESTING - make sure nothing sketchy happens at the breakpoint
		//out += "BREAKPOINT\n";

		//go again starting at posOnRef + lastM_ref using events starting at readHead + lastM_ev
		readHead += lastM_ev + 1;
		posOnRef += lastM_ref;
	}
	return out;
}


std::string eventalign_train( read &r,
            unsigned int windowLength,
			std::map<unsigned int, double> &BrdULikelihood){

	std::string out;
	//get the positions on the reference subsequence where we could attempt to make a call
	std::string strand;
	unsigned int readHead = 0;
	if ( r.isReverse ) strand = "rev";
	else strand = "fwd";

	out += ">" + r.readID + " " + r.referenceMappedTo + " " + std::to_string(r.refStart) + " " + std::to_string(r.refEnd) + " " + strand + "\n";

	unsigned int posOnRef = 0;
	while ( posOnRef < r.referenceSeqMappedTo.size() - windowLength - 7 ){ //-7 because we need to reach forward 6 bases when we process the match states

		std::string readSnippet = (r.referenceSeqMappedTo).substr(posOnRef, windowLength);

		//make sure the read snippet is fully defined as A/T/G/C in reference
		unsigned int As = 0, Ts = 0, Cs = 0, Gs = 0;
		for ( std::string::iterator i = readSnippet.begin(); i < readSnippet.end(); i++ ){

			switch( *i ){
				case 'A' :
					As++;
					break;
				case 'T' :
					Ts++;
					break;
				case 'G' :
					Gs++;
					break;
				case 'C' :
					Cs++;
					break;
			}
		}
		if ( readSnippet.length() != (As + Ts + Gs + Cs) ){
			continue;
		}
		std::vector< double > eventSnippet;
		std::vector< double > eventLengthsSnippet;

		//get events for the alignment
		for ( unsigned int j = readHead; j < (r.eventAlignment).size(); j++ ){

			/*if an event has been aligned to a position in the window, add it */
			if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef] and (r.eventAlignment)[j].second < (r.refToQuery)[posOnRef + windowLength] ){

				double ev = (r.normalisedEvents)[(r.eventAlignment)[j].first];
				if (ev > r.scalings.shift + 1.0 and ev < 250.0){
					eventSnippet.push_back( ev );
					eventLengthsSnippet.push_back( (r.eventLengths)[(r.eventAlignment)[j].first] );
				}
			}

			/*stop once we get to the end of the window */
			if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef + windowLength] ) break;
		}

		//pass on this window if we have a deletion
		if ( eventSnippet.size() < 2 ){

			posOnRef += windowLength;
			continue;
		}


		//calculate where we are on the assembly - if we're a reverse complement, we're moving backwards down the reference genome
		int globalPosOnRef;
		if ( r.isReverse ) globalPosOnRef = r.refEnd - posOnRef - 6;
		else globalPosOnRef = r.refStart + posOnRef;

		std::pair< double, std::vector< std::string > > localAlignment = eventViterbi( eventSnippet, readSnippet, r.scalings);

		std::vector< std::string > stateLabels = localAlignment.second;
		size_t lastM_ev = 0;
		size_t lastM_ref = 0;

		size_t evIdx = 0;

		//grab the index of the last match so we don't print insertions where we shouldn't
		for (size_t i = 1; i < stateLabels.size(); i++){

			assert(stateLabels[i] != "START");
			if (stateLabels[i] == "END") continue;

			std::string label = stateLabels[i].substr(stateLabels[i].find('_')+1);

	        if (label == "D") continue; //silent states don't emit an event

	        int pos = std::stoi(stateLabels[i].substr(0,stateLabels[i].find('_')));

	        if (label == "M"){
	        	lastM_ev = evIdx;
	        	lastM_ref = pos;
	        }
	        evIdx ++;
		}

		//do a second pass to print the alignment
		evIdx = 0;
		for (size_t i = 1; i < stateLabels.size(); i++){

			assert(stateLabels[i] != "START");
			if (stateLabels[i] == "END") continue;

			std::string label = stateLabels[i].substr(stateLabels[i].find('_')+1);

	        if (label == "D") continue; //silent states don't emit an event

	        int pos = std::stoi(stateLabels[i].substr(0,stateLabels[i].find('_')));
			std::string sixMerStrand = (r.referenceSeqMappedTo).substr(posOnRef + pos, 6);

			double scaledEvent = (eventSnippet[evIdx] - r.scalings.shift) / r.scalings.scale;
			double eventLength = eventLengthsSnippet[evIdx];

			assert(scaledEvent > 0.0);

			unsigned int evPos;
			std::string sixMerRef;
			if (r.isReverse){
				evPos = globalPosOnRef - pos;
				sixMerRef = reverseComplement(sixMerStrand);
			}
			else{
				evPos = globalPosOnRef + pos;
				sixMerRef = sixMerStrand;
			}

			if (label == "M" and BrdULikelihood.count(evPos) > 0){
				out += std::to_string(evPos) + "\t" + sixMerRef + "\t" + std::to_string(scaledEvent) + "\t" + std::to_string(eventLength) + "\t" + sixMerStrand + "\t" + std::to_string(thymidineModel.at(sixMerStrand).first) + "\t" + std::to_string(thymidineModel.at(sixMerStrand).second) + "\t" + std::to_string(BrdULikelihood[evPos]) + "\n";
			}
			else if (label == "M"){
				out += std::to_string(evPos) + "\t" + sixMerRef + "\t" + std::to_string(scaledEvent) + "\t" + std::to_string(eventLength) + "\t" + sixMerStrand + "\t" + std::to_string(thymidineModel.at(sixMerStrand).first) + "\t" + std::to_string(thymidineModel.at(sixMerStrand).second) + "\n";
			}
			else if (label == "I" and evIdx < lastM_ev){ //don't print insertions after the last match because we're going to align these in the next segment
				out += std::to_string(evPos) + "\t" + sixMerRef + "\t" + std::to_string(scaledEvent) + "\t" + std::to_string(eventLength) + "\t" + "NNNNNN" + "\t" + "0" + "\t" + "0" + "\n";
			}

	        evIdx ++;
		}

		//TESTING - make sure nothing sketchy happens at the breakpoint
		//out += "BREAKPOINT\n";

		//go again starting at posOnRef + lastM_ref using events starting at readHead + lastM_ev
		readHead += lastM_ev + 1;
		posOnRef += lastM_ref;
	}
	return out;
}


AlignedRead eventalign_detect( read &r,
            unsigned int windowLength ){

	std::string out;
	//get the positions on the reference subsequence where we could attempt to make a call
	std::string strand;
	unsigned int readHead = 0;
	if ( r.isReverse ) strand = "rev";
	else strand = "fwd";

	AlignedRead ar(r.readID, r.referenceMappedTo, strand, r.refStart, r.refEnd);

	unsigned int posOnRef = 0;
	while ( posOnRef < r.referenceSeqMappedTo.size() - windowLength - 7 ){ //-7 because we need to reach forward 6 bases when we process the match states

		std::string readSnippet = (r.referenceSeqMappedTo).substr(posOnRef, windowLength);

		//make sure the read snippet is fully defined as A/T/G/C in reference
		unsigned int As = 0, Ts = 0, Cs = 0, Gs = 0;
		for ( std::string::iterator i = readSnippet.begin(); i < readSnippet.end(); i++ ){

			switch( *i ){
				case 'A' :
					As++;
					break;
				case 'T' :
					Ts++;
					break;
				case 'G' :
					Gs++;
					break;
				case 'C' :
					Cs++;
					break;
			}
		}
		if ( readSnippet.length() != (As + Ts + Gs + Cs) ){
			continue;
		}
		std::vector< double > eventSnippet;
		std::vector< double > eventLengthsSnippet;

		/*get the events that correspond to the read snippet */
		for ( unsigned int j = readHead; j < (r.eventAlignment).size(); j++ ){

			/*if an event has been aligned to a position in the window, add it */
			if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef] and (r.eventAlignment)[j].second < (r.refToQuery)[posOnRef + windowLength] ){

				double ev = (r.normalisedEvents)[(r.eventAlignment)[j].first];
				if (ev > r.scalings.shift + 1.0 and ev < 250.0){
					eventSnippet.push_back( ev );
					eventLengthsSnippet.push_back( (r.eventLengths)[(r.eventAlignment)[j].first] );
				}
			}

			/*stop once we get to the end of the window */
			if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef + windowLength] ) break;
		}

		//pass on this window if we have a deletion
		if ( eventSnippet.size() < 2 ){

			posOnRef += windowLength;
			continue;
		}

		/*
		TESTING - print out the read snippet, the ONT model, and the aligned events
		std::cout << readSnippet << std::endl;
		for ( int pos = 0; pos < readSnippet.length()-5; pos++ ){

			std::cout << readSnippet.substr(pos,6) << "\t" << thymidineModel.at( readSnippet.substr(pos,6) ).first << std::endl;
		}
		for ( auto ev = eventSnippet.begin(); ev < eventSnippet.end(); ev++){
			double scaledEv =  (*ev - r.scalings.shift) / r.scalings.scale;
			std::cout << scaledEv << std::endl;
		}
		*/

		//calculate where we are on the assembly - if we're a reverse complement, we're moving backwards down the reference genome
		int globalPosOnRef;
		if ( r.isReverse ) globalPosOnRef = r.refEnd - posOnRef - 6;
		else globalPosOnRef = r.refStart + posOnRef;

		std::pair< double, std::vector< std::string > > localAlignment = eventViterbi( eventSnippet, readSnippet, r.scalings);

		std::vector< std::string > stateLabels = localAlignment.second;
		size_t lastM_ev = 0;
		size_t lastM_ref = 0;

		size_t evIdx = 0;
		for (size_t i = 1; i < stateLabels.size(); i++){

			assert(stateLabels[i] != "START");
			if (stateLabels[i] == "END") continue;

			std::string label = stateLabels[i].substr(stateLabels[i].find('_')+1);

	        if (label == "D") continue; //silent states don't emit an event

	        int pos = std::stoi(stateLabels[i].substr(0,stateLabels[i].find('_')));
			std::string sixMerStrand = (r.referenceSeqMappedTo).substr(posOnRef + pos, 6);

			double scaledEvent = (eventSnippet[evIdx] - r.scalings.shift) / r.scalings.scale;
			double eventLength = eventLengthsSnippet[evIdx];

			assert(scaledEvent > 0.0);

			size_t evPos;
			std::string sixMerRef;
			if (r.isReverse){
				evPos = globalPosOnRef - pos;
				sixMerRef = reverseComplement(sixMerStrand);
			}
			else{
				evPos = globalPosOnRef + pos;
				sixMerRef = sixMerStrand;
			}

			if (label == "M"){
				ar.addEvent(sixMerStrand, evPos, scaledEvent, eventLength);
	        	lastM_ev = evIdx;
	        	lastM_ref = pos;
	        }
	        evIdx ++;
		}

		//go again starting at posOnRef + lastM_ref using events starting at readHead + lastM_ev
		readHead += lastM_ev + 1;
		posOnRef += lastM_ref;
	}
	return ar;
}


int align_main( int argc, char** argv ){

	Arguments args = parseAlignArguments( argc, argv );
	bool bulkFast5;

	//load DNAscent index
	std::map< std::string, std::string > readID2path;
	parseIndex( args.indexFilename, readID2path, bulkFast5 );

	//import fasta reference
	std::map< std::string, std::string > reference = import_reference_pfasta( args.referenceFilename );

	std::ofstream outFile( args.outputFilename );
	if ( not outFile.is_open() ) throw IOerror( args.outputFilename );

	htsFile* bam_fh;
	hts_idx_t* bam_idx;
	bam_hdr_t* bam_hdr;
	hts_itr_t* itr;

	//load the bam
	std::cout << "Opening bam file... ";
	bam_fh = sam_open((args.bamFilename).c_str(), "r");
	if (bam_fh == NULL) throw IOerror(args.bamFilename);

	//load the index
	bam_idx = sam_index_load(bam_fh, (args.bamFilename).c_str());
	if (bam_idx == NULL) throw IOerror("index for "+args.bamFilename);

	//load the header
	bam_hdr = sam_hdr_read(bam_fh);
	std::cout << "ok." << std::endl;

	/*initialise progress */
	int numOfRecords = 0, prog = 0, failed = 0;
	countRecords( bam_fh, bam_idx, bam_hdr, numOfRecords, args.minQ, args.minL );
	progressBar pb(numOfRecords,true);

	//build an iterator for all reads in the bam file
	const char *allReads = ".";
	itr = sam_itr_querys(bam_idx,bam_hdr,allReads);

	unsigned int windowLength = 100;
	int result;
	int failedEvents = 0;
	unsigned int maxBufferSize;
	std::vector< bam1_t * > buffer;
	if ( args.threads <= 4 ) maxBufferSize = args.threads;
	else maxBufferSize = 4*(args.threads);

	do {
		//initialise the record and get the record from the file iterator
		bam1_t *record = bam_init1();
		result = sam_itr_next(bam_fh, itr, record);

		//add the record to the buffer if it passes the user's criteria, otherwise destroy it cleanly
		int mappingQual = record -> core.qual;
		int refStart,refEnd;
		getRefEnd(record,refStart,refEnd);
		int queryLen = record -> core.l_qseq;

		if ( mappingQual >= args.minQ and refEnd - refStart >= args.minL and queryLen != 0 ){

			buffer.push_back( record );
		}
		else{
			bam_destroy1(record);
		}

		/*if we've filled up the buffer with short reads, compute them in parallel */
		if (buffer.size() >= maxBufferSize or (buffer.size() > 0 and result == -1 ) ){

			#pragma omp parallel for schedule(dynamic) shared(buffer,windowLength,analogueModel,thymidineModel,methyl5mCModel,args,prog,failed) num_threads(args.threads)
			for (unsigned int i = 0; i < buffer.size(); i++){

				read r;

				//get the read name (which will be the ONT readID from Albacore basecall)
				const char *queryName = bam_get_qname(buffer[i]);
				if (queryName == NULL) continue;
				std::string s_queryName(queryName);
				r.readID = s_queryName;

				//iterate on the cigar string to fill up the reference-to-query coordinate map
				parseCigar(buffer[i], r.refToQuery, r.refStart, r.refEnd);

				//get the name of the reference mapped to
				std::string mappedTo(bam_hdr -> target_name[buffer[i] -> core.tid]);
				r.referenceMappedTo = mappedTo;

				//open fast5 and normalise events to pA
				r.filename = readID2path[s_queryName];

				/*get the subsequence of the reference this read mapped to */
				r.referenceSeqMappedTo = reference.at(r.referenceMappedTo).substr(r.refStart, r.refEnd - r.refStart);

				//fetch the basecall from the bam file
				r.basecall = getQuerySequence(buffer[i]);

				//account for reverse complements
				if ( bam_is_rev(buffer[i]) ){

					r.basecall = reverseComplement( r.basecall );
					r.referenceSeqMappedTo = reverseComplement( r.referenceSeqMappedTo );
					r.isReverse = true;
				}

				normaliseEvents(r, bulkFast5);

				//catch reads with rough event alignments that fail the QC
				if ( r.eventAlignment.size() == 0 ){

					failed++;
					prog++;
					continue;
				}

				std::string out = eventalign( r, windowLength);

				#pragma omp critical
				{
					outFile << out;
					prog++;
					pb.displayProgress( prog, failed, failedEvents );
				}
			}
			for ( unsigned int i = 0; i < buffer.size(); i++ ) bam_destroy1(buffer[i]);
			buffer.clear();
		}
		pb.displayProgress( prog, failed, failedEvents );
		if (args.capReads and prog > args.maxReads){
			sam_itr_destroy(itr);
			return 0;
		}
	} while (result > 0);
	sam_itr_destroy(itr);
	std::cout << std::endl;
	return 0;
}
