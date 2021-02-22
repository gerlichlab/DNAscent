//----------------------------------------------------------
// Copyright 2020 University of Cambridge
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#ifndef SENSE_H
#define SENSE_H

#include <cassert>
#include <vector>
#include "error_handling.h"
#include <iostream>

/*function prototypes */
int sense_main( int argc, char** argv );

class DetectedRead{

	public:
		std::vector< unsigned int > positions;
		std::vector< unsigned int > mappedPositions;
		std::vector< double > brduCalls;
		std::string readID, chromosome, strand, header;
		int mappingLower, mappingUpper;
		std::vector<std::vector<float>> probabilities;
		std::vector<std::pair<int,int>> origins;
		std::vector<std::pair<int,int>> terminations;
		std::vector<double> tensorInput;
		int64_t inputSize;
		void trim(unsigned int trimFactor){
			
			//assert(positions.size() > trimFactor and tensorInput.size() > trimFactor and positions.size() == tensorInput.size());

			/*			
			unsigned int cropFromEnd = tensorInput.size() % trimFactor;
			tensorInput.erase(tensorInput.end() - cropFromEnd, tensorInput.end());
			*/
			
			//positions.erase(positions.end() - cropFromEnd, positions.end());
			//assert(positions.size() % trimFactor == 0 and tensorInput.size() % trimFactor == 0);

			/*
			unsigned int cropFromEnd = (mappingUpper-mappingLower+1) % trimFactor;
			inputSize = (mappingUpper-mappingLower+1) - cropFromEnd;
			tensorInput.erase(tensorInput.end() - 2*cropFromEnd, tensorInput.end());

			unsigned int upperThymPos = mappingUpper - cropFromEnd;
			for (auto i = positions.size()-1; i >= 0; i--){

				if (positions[i] > upperThymPos) positions.pop_back();
				else break;
			}
			*/

			unsigned int cropFromEnd = mappedPositions.size() % trimFactor;
			inputSize = mappedPositions.size() - cropFromEnd;
			tensorInput.erase(tensorInput.end() - 2*cropFromEnd, tensorInput.end());
			mappedPositions.erase(mappedPositions.end() - cropFromEnd, mappedPositions.end());

			unsigned int upperThymPos = mappedPositions.back();
			for (auto i = positions.size()-1; i >= 0; i--){

				if (positions[i] > upperThymPos) positions.pop_back();
				else break;
			}


		}
		void generateInput(void){

			/*
			std::vector<double> out(mappingUpper-mappingLower+1,0.0001);
			for (size_t i = 0; i < positions.size(); i++){
				out[positions[i]-mappingLower] = brduCalls[i];
			}
			tensorInput=out;
			*/
			
			/*
			for (size_t i = 0; i < positions.size(); i++){
				tensorInput.push_back(brduCalls[i]);
			}
			*/
			

			/*
			int w = 501;
			int deg = 5;
			tensorInput = sg_smooth(brduCalls, w, deg);
			*/
			//adding moving average filter
			/*
			int window = 40;
			for (size_t i = 0; i < positions.size(); i++){

				double runningSum = 0;

				for (int j = -window/2; j < window/2; j++){

					if (i+j >= 0 and i+j < positions.size()) runningSum += brduCalls[i+j];

				}

				tensorInput.push_back(runningSum / (double) window);
			}
			*/
			/*

			int window = 10; //5;
			std::vector<double> out(2*(mappingUpper-mappingLower+1),0.);
			for (size_t i = 0; i < positions.size(); i++){

				ssize_t lowerIndex = -1;
				for (ssize_t j = i; j >= 0; j--){
									
					if (positions[i] - positions[j] > window) break;

					lowerIndex = j;
				}
				assert(lowerIndex != -1);

				ssize_t upperIndex = -1;
				for (ssize_t j = i; j < positions.size(); j++){
									
					if (positions[j] - positions[i] > window) break;

					upperIndex = j;
				}
				assert(upperIndex != -1);

				double runningSum = 0.;
				double count = 0.;
				for (size_t j = lowerIndex; j <= upperIndex; j++){
					runningSum += brduCalls[j];
					count += 1.;
				}
				out[2*(positions[i]-mappingLower)] = runningSum / count;
				out[2*(positions[i]-mappingLower)+1] = 1.;
			}
			tensorInput=out;
			*/

			int window = 10;
			std::vector<double> out(2*mappedPositions.size(),0.);
			int pIndex = 0;
			for (size_t p = 0; p < mappedPositions.size(); p++){

				//if it's a thymidine position
				if (mappedPositions[p] == positions[pIndex]){
					
					ssize_t lowerIndex = -1;
					for (ssize_t j = pIndex; j >= 0; j--){
										
						if (positions[pIndex] - positions[j] > window) break;

						lowerIndex = j;
					}
					assert(lowerIndex != -1);

					ssize_t upperIndex = -1;
					for (ssize_t j = pIndex; j < positions.size(); j++){
										
						if (positions[j] - positions[pIndex] > window) break;

						upperIndex = j;
					}
					assert(upperIndex != -1);

					double runningSum = 0.;
					double count = 0.;
					for (size_t j = lowerIndex; j <= upperIndex; j++){
						runningSum += brduCalls[j];
						count += 1.;
					}
					out[2*p] = runningSum / count;
					out[2*p+1] = 1.;
					pIndex++;
					
					if (pIndex >= positions.size()) break;
				}

			}
			tensorInput=out;
		}
};

#endif

