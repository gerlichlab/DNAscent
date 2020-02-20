//----------------------------------------------------------
// Copyright 2019 University of Oxford
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-2.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#include <fstream>
#include "annotate.h"
#include <math.h>
#include <stdlib.h>
#include <limits>
#include "common.h"
#include "data_IO.h"
#include "event_handling.h"
#include "probability.h"
#include "detect.h"
#include "../htslib/htslib/hts.h"
#include "../htslib/htslib/sam.h"
#include "../fast5/include/fast5.hpp"


static const char *help=
"annotate: DNAscent executable that labels training data for the Guppy flip-flop basecaller.\n"
"To run DNAscent annotate, do:\n"
"  ./DNAscent annotate [arguments]\n"
"Example:\n"
"  ./DNAscent annotate -b /path/to/alignment.bam -r /path/to/reference.fasta -i /path/to/index.index -o /path/to/output.out -t 20\n"
"Required arguments are:\n"
"  -b,--bam                  path to alignment BAM file,\n"
"  -r,--reference            path to genome reference in fasta format,\n"
"  -i,--index                path to DNAscent index,\n"
"  -o,--output               path to output file that will be generated.\n"
"Optional arguments are:\n"
"  -t,--threads              number of threads (default is 1 thread)\n"
"  --useReference            annotate the reference subsequence rather than the ONT basecall (default is ONT basecall),\n"
"  --llThreshold             log-likelihood threshold above which a 6mer is said to contain BrdU (default is 2.5),\n"
"  --methyl-aware            account for CpG, Dcm, and Dam methylation in BrdU calling,\n"
"  -q,--quality              minimum mapping quality (default is 20),\n"
"  --minLength               minimum read length in bp (default is 1000),\n"
"  --maxLength               minimum read length in bp (default is 10000).\n"
"Written by Michael Boemo, Department of Pathology, University of Cambridge.\n"
"Please submit bug reports to GitHub Issues (https://github.com/MBoemo/DNAscent/issues).";

struct Arguments {
	bool useReference;
	double callThreshold;
	std::string bamFilename;
	std::string referenceFilename;
	std::string outputFilename;
	std::string indexFilename;
	bool methylAware;
	double divergence;
	int minQ;
	int minL, maxL;
	unsigned int threads;
};

extern std::map< std::string, std::pair< double, double > > analogueModel;
extern std::map< std::string, std::pair< double, double > > thymidineModel;
extern std::map< std::string, std::pair< double, double > > methyl5mCModel;

Arguments parseAnnotateArguments( int argc, char** argv ){

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
	args.callThreshold = 2.5;
	args.minQ = 20;
	args.minL = 1000;
	args.maxL = 10000;
	args.methylAware = false;
	args.useReference = false;
	args.divergence = 2.0;

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
		else if ( flag == "--minLength" ){

			std::string strArg( argv[ i + 1 ] );
			args.minL = std::stoi( strArg.c_str() );
			i+=2;
		}
		else if ( flag == "--maxLength" ){

			std::string strArg( argv[ i + 1 ] );
			args.maxL = std::stoi( strArg.c_str() );
			i+=2;
		}
		else if ( flag == "--llThreshold" ){

			std::string strArg( argv[ i + 1 ] );
			args.callThreshold = std::stof( strArg.c_str() );
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
		else if ( flag == "--divergence" ){

			std::string strArg( argv[ i + 1 ] );
			args.divergence = std::stof(strArg.c_str());
			i+=2;
		}
		else if ( flag == "--useReference" ){

			args.useReference = true;
			i+=1;
		}
		else if ( flag == "--methyl-aware" ){

			args.methylAware = true;
			i+=1;
		}
		else throw InvalidOption( flag );
	}
	return args;
}


void countAnnotateRecords( htsFile *bam_fh, hts_idx_t *bam_idx, bam_hdr_t *bam_hdr, int &numOfRecords, int minQ, int minL, int maxL ){

	hts_itr_t* itr = sam_itr_querys(bam_idx,bam_hdr,".");
	int result;

	do {
		bam1_t *record = bam_init1();
		result = sam_itr_next(bam_fh, itr, record);
		if ( (record -> core.qual >= minQ) and (record -> core.l_qseq >= minL) and (record -> core.l_qseq <= maxL) ) numOfRecords++;
		bam_destroy1(record);
	} while (result > 0);

	//cleanup
	sam_itr_destroy(itr);
}


std::vector< int > annotatePosition( read &r, unsigned int windowLength, std::map< std::string, std::pair< double, double > > &analogueModel, double callThresh ){

	//get the positions on the reference subsequence where we could attempt to make a call
	std::vector< unsigned int > POIs = getPOIs( r.referenceSeqMappedTo, windowLength );

	std::vector< int > analoguePositions;

	for ( unsigned int i = 0; i < POIs.size(); i++ ){

		int posOnRef = POIs[i];
		//int posOnQuery = (r.refToQuery).at(posOnRef);

		std::string readSnippet = (r.referenceSeqMappedTo).substr(posOnRef - windowLength, 2*windowLength);

		//TESTING - print out the read snippet and the event and the ONT model
		//extern std::map< std::string, std::pair< double, double > > SixMer_model;
		//std::cout << readSnippet << std::endl;
		//for ( int pos = 0; pos < readSnippet.length()-5; pos++ ){
		
		//	std::cout << readSnippet.substr(pos,6) << "\t" << SixMer_model.at( readSnippet.substr(pos,6) ).first << std::endl;
		//}
		//END TESTING

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

		//catch spans with lots of insertions or deletions
		int spanOnQuery = (r.refToQuery)[posOnRef + windowLength] - (r.refToQuery)[posOnRef - windowLength];
		if ( spanOnQuery > 2.5*windowLength or spanOnQuery < 1.5*windowLength ) continue;

		/*get the events that correspond to the read snippet */
		//double alignmentScore = 0.0;
		for ( unsigned int j = 0; j < (r.eventAlignment).size(); j++ ){

			/*if an event has been aligned to a position in the window, add it */
			if ( (r.eventAlignment)[j].second >= (r.refToQuery)[posOnRef - windowLength] and (r.eventAlignment)[j].second < (r.refToQuery)[posOnRef + windowLength - 5] ){

				eventSnippet.push_back( (r.normalisedEvents)[(r.eventAlignment)[j].first] );

				//alignmentScore += r.posToScore[j];

				//TESTING - print the event snippet
				//double ev = (r.normalisedEvents)[(r.eventAlignment)[j].first];
				//std::cout << (ev - r.scalings.shift) / r.scalings.scale << std::endl;
				//END TESTING
			}

			/*stop once we get to the end of the window */
			if ( (r.eventAlignment)[j].second > (r.refToQuery)[posOnRef + windowLength] ) break;
		}
		//alignmentScore /= eventSnippet.size();

		//catch abnormally few or many events
		if ( eventSnippet.size() > 8*windowLength or eventSnippet.size() < windowLength ) continue;

		//figure out where the T's are
		//std::cout << ">--------------" << std::endl;
		std::string sixOI = (r.referenceSeqMappedTo).substr(posOnRef,6);
		//size_t BrdUStart = sixOI.find('T') + windowLength;
		//size_t BrdUEnd = sixOI.rfind('T') + windowLength;
		//std::cout << sixOI << std::endl;
		std::vector<double> BrdUscores;
		/*
		for ( unsigned int j = 0; j < sixOI.length(); j++ ){

			if ( sixOI.substr(j,1) == "T" ){
			
				double logProbAnalogue = sequenceProbability( eventSnippet, readSnippet, windowLength, true, analogueModel, r.scalings, j+windowLength );
				BrdUscores.push_back(logProbAnalogue);
				//std::cout << logProbAnalogue << std::endl;
			}
		}

		double BrdUmax = BrdUscores[0];
		for ( unsigned int j = 1; j < BrdUscores.size(); j++ ){

			if (BrdUscores[j] > BrdUmax) BrdUmax = BrdUscores[j];
		}
		
		//std::cout << "max: " << BrdUmax << std::endl;
		//double logProbAnalogue = sequenceProbability( eventSnippet, readSnippet, windowLength, true, analogueModel, r.scalings );
		double logProbThymidine = sequenceProbability( eventSnippet, readSnippet, windowLength, false, analogueModel, r.scalings, 0 );
		//std::cout << "thym: " << logProbThymidine << std::endl;
		double logLikelihoodRatio = BrdUmax - logProbThymidine;
		//double logLikelihoodRatio = logProbAnalogue - logProbThymidine;

		//note that everything thus far has been done in terms of the read, not the reference
		//in DNAscent detect, if you're on a read that mapped to the reverse strand, you have to reverse the coordinates at this point
		//we're just annotating reads, so we want to stay with 5' -> 3' reference to the reads, so don't do anything extra here

		if (logLikelihoodRatio >= callThresh ) analoguePositions.push_back(posOnRef);
		*/	
	}
	return analoguePositions;
}


std::string annotateBasecallFasta( std::map<unsigned int, unsigned int> &ref2query, std::string &basecall, std::vector<int> &callsOnRef ){

	std::vector<int> callsOnQuery;
	for ( auto i = callsOnRef.begin(); i < callsOnRef.end(); i++ ){

		int posOnQuery = ref2query[*i];
		callsOnQuery.push_back(posOnQuery);
	}
	std::sort(callsOnQuery.begin(), callsOnQuery.end());

	unsigned int readHead = 0;
	std::string annBasecall;
	for ( auto i = callsOnQuery.begin(); i < callsOnQuery.end(); i++ ){

		annBasecall += basecall.substr(readHead,*i - readHead);
		std::string annotated6mer;
		std::string natural6mer = basecall.substr(*i,6);
		for ( auto s = natural6mer.begin(); s < natural6mer.end(); s++ ){

			if (*s == 'T') annotated6mer += 'B';
			else annotated6mer += *s;
		}
		annBasecall += annotated6mer;
		readHead = *i + 6;
	}
	annBasecall += basecall.substr(readHead);

	return annBasecall;
}


std::string annotateReferenceFasta( std::string &basecall, std::vector<int> &callsOnRef ){

	unsigned int readHead = 0;
	std::string annBasecall;
	for ( auto i = callsOnRef.begin(); i < callsOnRef.end(); i++ ){

		annBasecall += basecall.substr(readHead,*i - readHead);
		std::string annotated6mer;
		std::string natural6mer = basecall.substr(*i,6);
		for ( auto s = natural6mer.begin(); s < natural6mer.end(); s++ ){

			if (*s == 'T') annotated6mer += 'B';
			else annotated6mer += *s;
		}
		annBasecall += annotated6mer;
		readHead = *i + 7;
	}
	annBasecall += basecall.substr(readHead);

	return annBasecall;
}


int annotate_main( int argc, char** argv ){

	std::cout << "DNAscent annotate is not supported yet." << std::endl;
	return 0;

	Arguments args = parseAnnotateArguments( argc, argv );
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
	bam_fh = sam_open((args.bamFilename).c_str(), "r");
	if (bam_fh == NULL) throw IOerror(args.bamFilename);

	//load the index
	bam_idx = sam_index_load(bam_fh, (args.bamFilename).c_str());
	if (bam_idx == NULL) throw IOerror("index for "+args.bamFilename);

	//load the header
	bam_hdr = sam_hdr_read(bam_fh);

	/*initialise progress */
	int numOfRecords = 0, prog = 0, failed = 0;
	countAnnotateRecords( bam_fh, bam_idx, bam_hdr, numOfRecords, args.minQ, args.minL, args.maxL );
	progressBar pb(numOfRecords,true);

	//build an iterator for all reads in the bam file
	const char *allReads = ".";
	itr = sam_itr_querys(bam_idx,bam_hdr,allReads);

	unsigned int windowLength = 10;
	int result;
	unsigned int maxBufferSize;
	std::vector< bam1_t * > buffer;
	if ( args.threads <= 4 ) maxBufferSize = args.threads;
	else maxBufferSize = 4*(args.threads);

	do {

		//initialise the record and get the record from the file iterator
		bam1_t *record = bam_init1();
		result = sam_itr_next(bam_fh, itr, record);

		int mappingQual = record -> core.qual;
		int queryLength = record -> core.l_qseq;

		//add the record to the buffer if it passes the user's criteria, otherwise destroy it cleanly
		if ( mappingQual >= args.minQ and queryLength >= args.minL and queryLength <= args.maxL ){
			buffer.push_back( record );
		}
		else{
			bam_destroy1(record);
		}
		
		/*if we've filled up the buffer with short reads, compute them in parallel */
		if (buffer.size() >= maxBufferSize or (buffer.size() > 0 and result == -1 ) ){

			#pragma omp parallel for schedule(dynamic) shared(buffer,windowLength,analogueModel,args,prog,failed) num_threads(args.threads)
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

				std::stringstream ss; 
				std::vector<int> callPositions = annotatePosition(r, windowLength, analogueModel, args.callThreshold);

				std::string annotatedBasecall;
				if (args.useReference) annotatedBasecall = annotateReferenceFasta(r.referenceSeqMappedTo, callPositions);
				else annotatedBasecall = annotateBasecallFasta(r.refToQuery, r.basecall, callPositions);

				ss << ">" << r.readID << std::endl << annotatedBasecall << std::endl;

				#pragma omp critical
				{
					outFile << ss.rdbuf();
					prog++;
					pb.displayProgress( prog, failed, 0 );
				}
			}
			for ( unsigned int i = 0; i < buffer.size(); i++ ) bam_destroy1(buffer[i]);
			buffer.clear();
		}
		pb.displayProgress( prog, failed, 0 );	
	} while (result > 0);
	sam_itr_destroy(itr);
	std::cout << std::endl;
	return 0;
}
