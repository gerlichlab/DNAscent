//----------------------------------------------------------
// Copyright 2019 University of Oxford
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

//#define TEST_HMM 1
//#define TEST_LL 1
//#define TEST_ALIGNMENT 1
//#define TEST_METHYL 1

#include <fstream>
#include "detect.h"
#include <math.h>
#include <stdlib.h>
#include <limits>
#include "common.h"
#include "event_handling.h"
#include "probability.h"
#include "../fast5/include/fast5.hpp"
#include "poreModels.h"
#include "detect.h"

static const char *help=
"trainingData: DNAscent executable that generates training data for DNAscent LSTM analogue detection.\n"
"To run DNAscent trainingData, do:\n"
		"  ./DNAscent detect -b /path/to/alignment.bam -r /path/to/reference.fasta -i /path/to/index.dnascent -o /path/to/output.detect\n"
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

Arguments parseDataArguments( int argc, char** argv ){

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


std::map<int, std::vector<std::pair<double,double>>> referenceToEvents( read &r){

	std::map<int, std::vector<std::pair<double,double>>> composedMaps;

	for (size_t i = 0; i != r.refToQuery.size(); i++){

		size_t posOnRef = i;
		size_t posOnQuery = r.refToQuery[i];
		for (size_t j = 0; j < r.eventAlignment.size(); j++){

			if (r.eventAlignment[j].second == posOnQuery){

				composedMaps[posOnRef].push_back(std::make_pair(r.normalisedEvents[r.eventAlignment[j].first], r.eventLengths[r.eventAlignment[j].first]));
			}
			if (r.eventAlignment[j].second > posOnQuery) break;
		}
	}
	return composedMaps;
}


std::string getAlignedEvents(read &r, unsigned int windowLength){

	std::string out;
	//get the positions on the reference subsequence where we could attempt to make a call
	std::string strand;
	if ( r.isReverse ) strand = "rev";
	else strand = "fwd";

	std::map<int, std::vector<std::pair<double,double>>> composedMaps = referenceToEvents( r);

	out += ">" + r.readID + " " + r.referenceMappedTo + " " + std::to_string(r.refStart) + " " + std::to_string(r.refEnd) + " " + strand + "\n";

	for (unsigned int posOnRef = 2*windowLength; posOnRef < r.referenceSeqMappedTo.size() - 2*windowLength; posOnRef++){

		if (composedMaps.count(posOnRef) > 0){

			std::string callLL = "";
			//decide if we're going to make a call here
			if ((r.referenceSeqMappedTo).substr(posOnRef,1) == "T"){
				std::string readSnippet = (r.referenceSeqMappedTo).substr(posOnRef - windowLength, 2*windowLength+6);

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
				if ( readSnippet.length() != (As + Ts + Gs + Cs) ) continue;

				std::vector< double > eventSnippet;
				unsigned int readHead = 0;
				bool first = true;
				for ( unsigned int j = readHead; j < (r.eventAlignment).size(); j++ ){

					/*if an event has been aligned to a position in the window, add it */
					if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef - windowLength] and (r.eventAlignment)[j].second < (r.refToQuery)[posOnRef + windowLength] ){

						if (first){
							readHead = j;
							first = false;
							//std::cout << "READHEAD:" << j << " " << readHead << std::endl;
						}

						double ev = (r.normalisedEvents)[(r.eventAlignment)[j].first];
						if (ev > 0 and ev < 250){
							eventSnippet.push_back( ev );
						}
					}

					/*stop once we get to the end of the window */
					if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef + windowLength] ) break;
				}

				std::string sixOI = (r.referenceSeqMappedTo).substr(posOnRef,6);
				size_t BrdUStart = sixOI.find('T') + windowLength - 5;
				size_t BrdUEnd = windowLength;//sixOI.rfind('T') + windowLength;
				double logProbAnalogue = sequenceProbability( eventSnippet, readSnippet, windowLength, true, r.scalings, BrdUStart, BrdUEnd );
				double logProbThymidine = sequenceProbability( eventSnippet, readSnippet, windowLength, false, r.scalings, 0, 0 );
				double logLikelihoodRatio = logProbAnalogue - logProbThymidine;
				callLL = std::to_string(logLikelihoodRatio);
			}

			for (auto e = composedMaps[posOnRef].begin(); e < composedMaps[posOnRef].end(); e++){

				double scaledEvent = (e -> first - r.scalings.shift) / r.scalings.scale;
				std::string sixMer = r.referenceSeqMappedTo.substr(posOnRef,6);

				out += std::to_string(posOnRef) + "\t" + sixMer + "\t" + std::to_string(scaledEvent) + "\t" + std::to_string(e -> second) + "\t" + std::to_string(thymidineModel.at(sixMer).first) + "\t" + std::to_string(thymidineModel.at(sixMer).second) + "\t" + callLL + "\n";  //reference 6mer, event, event length
			}
		}
		else{

			out += "NNNNNN\t0\t0\n";  //reference 6mer, event, event length
		}
	}
	return out;
}


int data_main( int argc, char** argv ){

	Arguments args = parseDataArguments( argc, argv );
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

	unsigned int windowLength = 10;
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

				try{

					if (bulkFast5) bulk_getEvents(r.filename, r.readID, r.raw);			
					else getEvents( r.filename, r.raw );
				}
				catch ( BadFast5Field &bf5 ){

					failed++;
					prog++;
					continue;
				}
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

				normaliseEvents(r);

				//catch reads with rough event alignments that fail the QC
				if ( r.eventAlignment.size() == 0 ){

					failed++;
					prog++;
					continue;
				}

				std::string readOut = getAlignedEvents(r, windowLength);

				#pragma omp critical
				{
					outFile << readOut;
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
