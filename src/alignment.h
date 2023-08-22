//----------------------------------------------------------
// Copyright 2019-2020 University of Oxford
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#ifndef ALIGN_H
#define ALIGN_H

#include <fstream>
#include "detect.h"
#include <math.h>
#include <stdlib.h>
#include <limits>
#include "common.h"
#include "event_handling.h"
#include "../fast5/include/fast5.hpp"
#include "common.h"
#include <memory>
#include <utility>
#include "config.h"

#define NFEATURES 5
#define RAWDEPTH 20

class AlignedPosition{

	private:

		bool forTraining = false;
		std::string kmer;
		unsigned int refPos;
		std::vector<double> signal;
		double eventAlignQuality;

	public:
		AlignedPosition(std::string kmer, unsigned int refPos, int quality){

			this -> kmer = kmer;
			this -> refPos = refPos;
			this -> eventAlignQuality = quality;
		}
		~AlignedPosition() {};
		void addSignal(double s){

			signal.push_back(s);
		}
		std::string getKmer(void){

			return kmer;
		}
		unsigned int getRefPos(void){

			return refPos;
		}
		double getAlignmentQuality(void){
			
			return eventAlignQuality;
		}
		std::vector<float> makeSignalFeature(void){

			assert(signal.size() > 0);
			assert(kmer.substr(0,1) == "A" || kmer.substr(0,1) == "T" || kmer.substr(0,1) == "G" || kmer.substr(0,1) == "C");

			std::vector<float> padded_signal;

			for (size_t i = 0; i < signal.size(); i++){

				padded_signal.push_back(signal[i]);

				if (i == RAWDEPTH - 1) break;
			}
			
			//zero padding if this position has few raw events - these will be masked by the neural network
			if (signal.size() < RAWDEPTH){

				for (size_t i = 0; i < RAWDEPTH - signal.size(); i++){

					padded_signal.push_back(0.);
				}
			}

			assert(padded_signal.size() == RAWDEPTH);
			return padded_signal;
		}
		std::vector<float> makeSequenceFeature(void){

			assert(signal.size() > 0);
			assert(kmer.substr(0,1) == "A" || kmer.substr(0,1) == "T" || kmer.substr(0,1) == "G" || kmer.substr(0,1) == "C");
			
			//one-hot encode bases
			std::vector<float> feature = {0., 0., 0., 0.};
			if (kmer.substr(0,1) == "A") feature[0] = 1.;
			else if (kmer.substr(0,1) == "T") feature[1] = 1.;
			else if (kmer.substr(0,1) == "G") feature[2] = 1.;
			else if (kmer.substr(0,1) == "C") feature[3] = 1.;

			//expected signal
			std::pair<double,double> meanStd = Pore_Substrate_Config.pore_model[kmer2index(kmer, Pore_Substrate_Config.kmer_len)];
			feature.push_back(meanStd.first);

			return feature;
		}
};


class AlignedRead{

	private:
		std::string readID, chromosome, strand;
		std::map<unsigned int, std::shared_ptr<AlignedPosition>> positions;
		unsigned int mappingLower, mappingUpper;
		std::vector<double> alignmentQualities;

	public:
		bool QCpassed;
		std::string stdout;
	
		AlignedRead(std::string readID, std::string chromosome, std::string strand, unsigned int ml, unsigned int mu, unsigned int numEvents){

			this -> readID = readID;
			this -> chromosome = chromosome;
			this -> strand = strand;
			this -> mappingLower = ml;
			this -> mappingUpper = mu;
			this -> QCpassed = false;
		}
		AlignedRead( const AlignedRead &ar ){

			this -> readID = ar.readID;
			this -> chromosome = ar.chromosome;
			this -> strand = ar.strand;
			this -> mappingLower = ar.mappingLower;
			this -> mappingUpper = ar.mappingUpper;
			this -> positions = ar.positions;
		}
		~AlignedRead(){}
		void addSignal(std::string kmer, unsigned int refPos, double sig, int quality){

			if (positions.count(refPos) == 0){

				std::shared_ptr<AlignedPosition> ap( new AlignedPosition(kmer, refPos, quality));
				ap -> addSignal(sig);
				positions[refPos] = ap;
			}
			else{

				positions[refPos] -> addSignal(sig);
			}
		}
		std::string getReadID(void){
			return readID;
		}
		std::string getChromosome(void){
			return chromosome;
		}
		std::string getStrand(void){
			return strand;
		}
		unsigned int getMappingLower(void){
			return mappingLower;
		}
		unsigned int getMappingUpper(void){
			return mappingUpper;
		}
		std::vector<float> makeSignalTensor(void){

			assert(strand == "fwd" || strand == "rev");
			std::vector<float> tensor;
			tensor.reserve(RAWDEPTH * positions.size());

			if (strand == "fwd"){

				for (auto p = positions.begin(); p != positions.end(); p++){

					std::vector<float> feature = (p -> second) -> makeSignalFeature();
					tensor.insert(tensor.end(), feature.begin(), feature.end());
				}
			}
			else{

				for (auto p = positions.rbegin(); p != positions.rend(); p++){

					std::vector<float> feature = (p -> second) -> makeSignalFeature();
					tensor.insert(tensor.end(), feature.begin(), feature.end());
				}
			}
			return tensor;
		}
		std::vector<float> makeSequenceTensor(void){

			assert(strand == "fwd" || strand == "rev");
			std::vector<float> tensor;
			tensor.reserve(NFEATURES * positions.size());

			if (strand == "fwd"){

				for (auto p = positions.begin(); p != positions.end(); p++){

					std::vector<float> feature = (p -> second) -> makeSequenceFeature();
					tensor.insert(tensor.end(), feature.begin(), feature.end());
				}
			}
			else{

				for (auto p = positions.rbegin(); p != positions.rend(); p++){

					std::vector<float> feature = (p -> second) -> makeSequenceFeature();
					tensor.insert(tensor.end(), feature.begin(), feature.end());
				}
			}
			return tensor;
		}
		std::vector<unsigned int> getPositions(void){

			std::vector<unsigned int> out;
			out.reserve(positions.size());
			if (strand == "fwd"){

				for (auto p = positions.begin(); p != positions.end(); p++){
					out.push_back(p -> first);
				}
			}
			else{

				for (auto p = positions.rbegin(); p != positions.rend(); p++){
					out.push_back(p -> first);
				}
			}
			return out;
		}
		std::vector<int> getAlignmentQuality(void){

			std::vector<int> out;
			out.reserve(positions.size());
			if (strand == "fwd"){

				for (auto p = positions.begin(); p != positions.end(); p++){
					out.push_back( (p -> second) -> getAlignmentQuality() );
				}
			}
			else{

				for (auto p = positions.rbegin(); p != positions.rend(); p++){
					out.push_back( (p -> second) -> getAlignmentQuality() );
				}
			}
			return out;
		}
		std::string getCigar(void){

			std::vector<std::pair<int,int>> cigarTuples;
			if (strand == "fwd"){

				int matchRun = 0;

				for (auto p = std::next(positions.begin()); p != positions.end(); p++){

					//deletion
					if ((p -> first) - (std::prev(p) -> first) != 1){

						if (matchRun > 0) cigarTuples.push_back(std::make_pair(1,matchRun));
						matchRun = 0;
						cigarTuples.push_back(std::make_pair(0,(p -> first) - (std::prev(p) -> first)-1));
					}
					else{
						matchRun++;
					}
				}
				if (matchRun > 0) cigarTuples.push_back(std::make_pair(1,matchRun));

			}
			else{

				int matchRun = 0;

				for (auto p = std::next(positions.rbegin()); p != positions.rend(); p++){

					//deletion
					if ((p -> first) - (std::prev(p) -> first) != 1){

						if (matchRun > 0) cigarTuples.push_back(std::make_pair(1,matchRun));
						matchRun = 0;
						cigarTuples.push_back(std::make_pair(0,(p -> first) - (std::prev(p) -> first)-1));
					}
					else{
						matchRun++;
					}
				}
				if (matchRun > 0) cigarTuples.push_back(std::make_pair(1,matchRun));
			}

			//convert to string
			std::string cigarString;
			for (auto c = cigarTuples.begin(); c < cigarTuples.end(); c++){

				//deletion
				if (c -> first == 0) cigarString += std::to_string(c -> second) + "D";

				//match
				if (c -> first == 1) cigarString +=  std::to_string(c -> second) + "M";
			}

			return cigarString;
		}
		std::vector<std::string> getKmers(void){

			std::vector<std::string> out;
			out.reserve(positions.size());
			if (strand == "fwd"){

				for (auto p = positions.begin(); p != positions.end(); p++){
					out.push_back((p -> second) -> getKmer());
				}
			}
			else{

				for (auto p = positions.rbegin(); p != positions.rend(); p++){
					out.push_back((p -> second) -> getKmer());
				}
			}
			return out;
		}
		std::vector<size_t> getSignalShape(void){

			return {positions.size(), RAWDEPTH, 1};
		}
		std::vector<size_t> getSequenceShape(void){

			return {positions.size(), NFEATURES};
		}
};


/*function prototypes */
int align_main( int argc, char** argv );
std::shared_ptr<AlignedRead> eventalign( read &, unsigned int , std::map<unsigned int,std::pair<double,double>> &);

#endif
