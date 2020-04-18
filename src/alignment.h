//----------------------------------------------------------
// Copyright 2020 University of Cambridge
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
#include "poreModels.h"
#include "common.h"
#include <memory>


class AlignedPosition{

	private:

		bool forTraining = false;
		std::string sixMer;
		unsigned int refPos;
		std::vector<double> events;
		std::vector<double> lengths;

	public:
		AlignedPosition(std::string sixMer, unsigned int refPos){

			this -> sixMer = sixMer;
			this -> refPos = refPos;
		}
		~AlignedPosition() {};
		void addEvent(double ev, double len){

			events.push_back(ev);
			lengths.push_back(len);
		}
		std::vector<double> makeFeature(void){

			assert(events.size() > 0 && events.size() == lengths.size());
			assert(sixMer.substr(0,1) == "A" || sixMer.substr(0,1) == "T" || sixMer.substr(0,1) == "G" || sixMer.substr(0,1) == "C" || sixMer.substr(0,1) == "N");

			//one-hot encode bases
			std::vector<double> feature = {0., 0., 0., 0., 0.};
			if (sixMer.substr(0,1) == "A") feature[0] = 1.;
			else if (sixMer.substr(0,1) == "T") feature[1] = 1.;
			else if (sixMer.substr(0,1) == "G") feature[2] = 1.;
			else if (sixMer.substr(0,1) == "C") feature[3] = 1.;
			else if (sixMer.substr(0,1) == "N") feature[4] = 1.;

			//events
			double eventMean = vectorMean(events);
			feature.push_back(eventMean);
			feature.push_back(vectorStdv(events, eventMean));

			//stutter
			feature.push_back((double)events.size());

			//event lengths
			double lengthsMean = vectorMean(lengths);
			feature.push_back(lengthsMean);
			feature.push_back(vectorStdv(lengths, lengthsMean));

			//pore model
			feature.push_back(thymidineModel.at(sixMer).first);
			feature.push_back(thymidineModel.at(sixMer).second);

			return feature;
		}


};


class AlignedRead{

	private:
		std::string readID, chromosome, strand;
		std::map<unsigned int, std::shared_ptr<AlignedPosition>> positions;
		unsigned int mappingLower, mappingUpper;

	public:
		AlignedRead(std::string readID, std::string chromosome, std::string strand, unsigned int ml, unsigned int mu){

			this -> readID = readID;
			this -> chromosome = chromosome;
			this -> strand = strand;
			this -> mappingLower = ml;
			this -> mappingUpper = mu;
		}
		~AlignedRead(){}
		void addEvent(std::string sixMer, unsigned int refPos, double ev, double len){

			if (positions.count(refPos) == 0){

				std::shared_ptr<AlignedPosition> ap( new AlignedPosition(sixMer, refPos));
				ap -> addEvent(ev,len);
				positions[refPos] = ap;
			}
			else{

				positions[refPos] -> addEvent(ev,len);
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
		std::vector<std::vector<double>> makeTensor(void){

			assert(strand == "fwd" || strand == "rev");
			std::vector<std::vector<double>> tensor;

			if (strand == "fwd"){

				for (auto p = positions.begin(); p != positions.end(); p++){

					//sort out gaps
					double gap = 0.0;
					if (p != positions.begin()) gap = (p -> first) - (std::prev(p) -> first);

					std::vector<double> feature = (p -> second) -> makeFeature();
					feature.push_back(gap);
					tensor.push_back(feature);
				}
			}
			else{

				for (auto p = positions.rbegin(); p != positions.rend(); p++){

					//sort out gaps
					double gap = 0.0;
					if (p != positions.rbegin()) gap =  (std::prev(p) -> first) - (p -> first);

					std::vector<double> feature = (p -> second) -> makeFeature();
					feature.push_back(gap);
					tensor.push_back(feature);
				}
			}
			return tensor;
		}
};



/*function prototypes */
int align_main( int argc, char** argv );
std::string eventalign_train( read &, unsigned int , std::map<unsigned int, double> &);

#endif
