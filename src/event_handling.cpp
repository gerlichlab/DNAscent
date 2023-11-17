//----------------------------------------------------------
// Copyright 2019-2020 University of Oxford
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

//#define EVENT_LENGTHS 1
//#define SHOW_PROGRESS 1

#include <iterator>
#include <algorithm>
#include <math.h>
#include "probability.h"
#include "error_handling.h"
#include "event_handling.h"
#include "../fast5/include/fast5.hpp"
#include <chrono>
#include "config.h"

#include "scrappie/event_detection.h"
#include "scrappie/scrappie_common.h"

// #include "../pod5-file-format/c++/pod5_format/c_api.h"


#define _USE_MATH_DEFINES

//from scrappie
float fast5_read_float_attribute(hid_t group, const char *attribute) {
	float val = NAN;
	if (group < 0) {
#ifdef DEBUG_FAST5_IO
		fprintf(stderr, "Invalid group passed to %s:%d.", __FILE__, __LINE__);
#endif
		return val;
	}

	hid_t attr = H5Aopen(group, attribute, H5P_DEFAULT);
	if (attr < 0) {
#ifdef DEBUG_FAST5_IO
		fprintf(stderr, "Failed to open attribute '%s' for reading.", attribute);
#endif
		return val;
	}

	H5Aread(attr, H5T_NATIVE_FLOAT, &val);
	H5Aclose(attr);

	return val;
}
//end scrappie


void fast5_getSignal( read &r ){

	//open the file
	hid_t hdf5_file = H5Fopen(r.filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
	if (hdf5_file < 0) throw IOerror(r.filename.c_str());

	//get the channel parameters
	std::string scaling_path = "/read_" + r.readID + "/channel_id";
	hid_t scaling_group = H5Gopen(hdf5_file, scaling_path.c_str(), H5P_DEFAULT);
	float digitisation = fast5_read_float_attribute(scaling_group, "digitisation");
	float offset = fast5_read_float_attribute(scaling_group, "offset");
	float range = fast5_read_float_attribute(scaling_group, "range");
	//float sample_rate = fast5_read_float_attribute(scaling_group, "sampling_rate");
	H5Gclose(scaling_group);

	//get the raw signal
	hid_t space;
	hsize_t nsample;
	float raw_unit;
	float *rawptr = NULL;

	std::string signal_path = "/read_" + r.readID + "/Raw/Signal";
	hid_t dset = H5Dopen(hdf5_file, signal_path.c_str(), H5P_DEFAULT);
	if (dset < 0 ) throw BadFast5Field(); 
	space = H5Dget_space(dset);
	if (space < 0 ) throw BadFast5Field(); 
	H5Sget_simple_extent_dims(space, &nsample, NULL);
   	rawptr = (float*)calloc(nsample, sizeof(float));
    	herr_t status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, rawptr);

	if ( status < 0 ){
		free(rawptr);
		H5Dclose(dset);
		return;
	}
	H5Dclose(dset);
	
	raw_unit = range / digitisation;
	r.raw.reserve(nsample);
	for ( size_t i = 0; i < nsample; i++ ){

		r.raw.push_back( (rawptr[i] + offset) * raw_unit );
	}

	free(rawptr);
	H5Fclose(hdf5_file);
}

/*
void pod5_getSignal( read &r ){

	Pod5FileReader_t *pod5_file = pod5_open_file(r.filename.c_str());
	if (pod5_file < 0) throw IOerror(r.filename.c_str());

	size_t batch_count = 0;
	if (pod5_get_read_batch_count(&batch_count, pod5_file) != POD5_OK) {
		std::cerr << "Failed to query batch count: " << pod5_get_error_string() << "\n";
		return EXIT_FAILURE;
	}
	
	size_t read_count = 0;

	for (std::size_t batch_index = 0; batch_index < batch_count; ++batch_index) {
		std::cout << "batch_index: " << batch_index + 1 << "/" << batch_count << "\n";

		Pod5ReadRecordBatch_t * batch = nullptr;
		if (pod5_get_read_batch(&batch, file, batch_index) != POD5_OK) {
			std::cerr << "Failed to get batch: " << pod5_get_error_string() << "\n";
			return EXIT_FAILURE;
		}

		std::size_t batch_row_count = 0;
		if (pod5_get_read_batch_row_count(&batch_row_count, batch) != POD5_OK) {
			std::cerr << "Failed to get batch row count\n";
			return EXIT_FAILURE;
		}

		for (std::size_t row = 0; row < batch_row_count; ++row) {
			uint16_t read_table_version = 0;
			ReadBatchRowInfo_t read_data;
			if (pod5_get_read_batch_row_info_data(
				batch, row, READ_BATCH_ROW_INFO_VERSION, &read_data, &read_table_version)
				!= POD5_OK)
			{
				std::cerr << "Failed to get read " << row << "\n";
				return EXIT_FAILURE;
			}

			read_count += 1;

			std::size_t sample_count = 0;
			pod5_get_read_complete_sample_count(file, batch, row, &sample_count);

			std::vector<std::int16_t> samples;
			samples.resize(sample_count);
			pod5_get_read_complete_signal(file, batch, row, samples.size(), samples.data());

			// Run info
			RunInfoDictData_t * run_info = nullptr;
			if (pod5_get_run_info(batch, read_data.run_info, &run_info) != POD5_OK) {
				throw std::runtime_error(
					"Failed to get run info " + std::to_string(read_data.run_info) + " : "
					+ pod5_get_error_string());
			}

			pod5_free_run_info(run_info);
		}

		if (pod5_free_read_batch(batch) != POD5_OK) {
			std::cerr << "Failed to release batch\n";
			return EXIT_FAILURE;
		}
	}

	std::cout << "Extracted " << read_count << " reads " << "\n";

	// Close the reader
	if (pod5_close_and_free_reader(file) != POD5_OK) {
		std::cerr << "Failed to close reader: " << pod5_get_error_string() << "\n";
		return EXIT_FAILURE;
	}

	// Cleanup the library
	pod5_terminate();
}
*/


//start: adapted from nanopolish (https://github.com/jts/nanopolish)
//licensed under MIT

inline float logProbabilityMatch(unsigned int kmerIndex, event e, double shift, double scale){

	std::pair<double,double> meanStd = Pore_Substrate_Config.pore_model[kmerIndex]; 
	double mu = meanStd.first;
	double sigma = meanStd.second;
	
	//scale the signal to the pore model
	double x = (e.mean - shift)/scale;

	//cauchy distribution
	//float a = (x - mu) / sigma;	
	//double thymProb = -eln(M_PI * sigma) - eln(1 + a * a);
	
	//normal distribution
	//float a = (x - mu) / 0.24;	
	float a = (x - mu) / sigma;	
	static const float log_inv_sqrt_2pi = log(0.3989422804014327);
	//double thymProb = log_inv_sqrt_2pi - eln(0.24) + (-0.5f * a * a);
	double thymProb = log_inv_sqrt_2pi - eln(sigma) + (-0.5f * a * a);
	return thymProb;
}	

#define event_kmer_to_band(ei, ki) (ei + 1) + (ki + 1)
#define band_event_to_offset(bi, ei) band_lower_left[bi].event_idx - (ei)
#define band_kmer_to_offset(bi, ki) (ki) - band_lower_left[bi].kmer_idx
#define is_offset_valid(offset) (offset) >= 0 && (offset) < bandwidth
#define event_at_offset(bi, offset) band_lower_left[(bi)].event_idx - (offset)
#define kmer_at_offset(bi, offset) band_lower_left[(bi)].kmer_idx + (offset)
#define move_down(curr_band) { curr_band.event_idx + 1, curr_band.kmer_idx }
#define move_right(curr_band) { curr_band.event_idx, curr_band.kmer_idx + 1 }

void adaptive_banded_simple_event_align( read &r, PoreParameters &s, std::vector<unsigned int> &kmer_ranks ){

	//benchmarking
	//std::chrono::steady_clock::time_point tp1 = std::chrono::steady_clock::now();

	std::string sequence = r.basecall;

	size_t k = Pore_Substrate_Config.kmer_len;
	size_t n_events = r.events.size();
	size_t n_kmers = sequence.size() - k + 1;

	// backtrack markers
	const uint8_t FROM_D = 0;
	const uint8_t FROM_U = 1;
	const uint8_t FROM_L = 2;
 
	// qc
	double min_average_log_emission = Pore_Substrate_Config.AdaptiveBanded_config.min_average_log_emission;
	int max_gap_threshold = Pore_Substrate_Config.AdaptiveBanded_config.max_gap_threshold;

	// banding
	int bandwidth = Pore_Substrate_Config.AdaptiveBanded_config.bandwidth;

	int half_bandwidth = bandwidth / 2;
 
	// transition penalties
	double events_per_kmer = (double)n_events / n_kmers;
	double p_stay = 1 - (1 / (events_per_kmer + 1));

	// setting a tiny skip penalty helps keep the true alignment within the adaptive band
	// this was empirically determined
	double epsilon = 1e-30;
	double lp_skip = log(epsilon);
	double lp_stay = log(p_stay);
	double lp_step = log(1.0 - exp(lp_skip) - exp(lp_stay));
	double lp_trim = log(0.01);
 
	// dp matrix
	size_t n_rows = n_events + 1;
	size_t n_cols = n_kmers + 1;
	size_t n_bands = n_rows + n_cols;
 
	// Initialize
	typedef std::vector<float> bandscore;
	typedef std::vector<uint8_t> bandtrace;

	std::vector<bandscore> bands(n_bands);
	std::vector<bandtrace> trace(n_bands);

	for(size_t i = 0; i < n_bands; ++i) {
		bands[i].resize(bandwidth, -INFINITY);
		trace[i].resize(bandwidth, 0);
	}

	// Keep track of the event/kmer index for the lower left corner of the band
	// these indices are updated at every iteration to perform the adaptive banding
	// Only the first two bands have their coordinates initialized, the rest are computed adaptively
	struct EventKmerPair {
		int event_idx;
		int kmer_idx;
	};

	std::vector<EventKmerPair> band_lower_left(n_bands);
 
	// initialize range of first two bands
	band_lower_left[0].event_idx = half_bandwidth - 1;
	band_lower_left[0].kmer_idx = -1 - half_bandwidth;
	band_lower_left[1] = move_down(band_lower_left[0]);

	// band 0: score zero in the central cell
	int start_cell_offset = band_kmer_to_offset(0, -1);
	assert(is_offset_valid(start_cell_offset));
	assert(band_event_to_offset(0, -1) == start_cell_offset);
	bands[0][start_cell_offset] = 0.0f;
    
	// band 1: first event is trimmed
	int first_trim_offset = band_event_to_offset(1, 0);
	assert(kmer_at_offset(1, first_trim_offset) == -1);
	assert(is_offset_valid(first_trim_offset));
	bands[1][first_trim_offset] = lp_trim;
	trace[1][first_trim_offset] = FROM_U;

	int fills = 0;

	// fill in remaining bands
	for(unsigned int band_idx = 2; band_idx < n_bands; ++band_idx) {
	// Determine placement of this band according to Suzuki's adaptive algorithm
        // When both ll and ur are out-of-band (ob) we alternate movements
        // otherwise we decide based on scores
		float ll = bands[band_idx - 1][0];
		float ur = bands[band_idx - 1][bandwidth - 1];
		bool ll_ob = ll == -INFINITY;
		bool ur_ob = ur == -INFINITY;
        
		bool right = false;
		if(ll_ob && ur_ob) {
			right = band_idx % 2 == 1;
		} else {
			right = ll < ur; // Suzuki's rule
		}

		if(right) {
			band_lower_left[band_idx] = move_right(band_lower_left[band_idx - 1]);
		} else {
			band_lower_left[band_idx] = move_down(band_lower_left[band_idx - 1]);
		}

		// If the trim state is within the band, fill it in here
		int trim_offset = band_kmer_to_offset(band_idx, -1);
		if(is_offset_valid(trim_offset)) {
			unsigned int event_idx = event_at_offset(band_idx, trim_offset);
			if(event_idx >= 0 && event_idx < n_events) {
				bands[band_idx][trim_offset] = lp_trim * (event_idx + 1);
				trace[band_idx][trim_offset] = FROM_U;
			} else {
				bands[band_idx][trim_offset] = -INFINITY;
			}
		}

		// Get the offsets for the first and last event and kmer
		// We restrict the inner loop to only these values
		int kmer_min_offset = band_kmer_to_offset(band_idx, 0);
		int kmer_max_offset = band_kmer_to_offset(band_idx, n_kmers);
		int event_min_offset = band_event_to_offset(band_idx, n_events - 1);
		int event_max_offset = band_event_to_offset(band_idx, -1);

		int min_offset = std::max(kmer_min_offset, event_min_offset);
		min_offset = std::max(min_offset, 0);

		int max_offset = std::min(kmer_max_offset, event_max_offset);
		max_offset = std::min(max_offset, bandwidth);

		for(int offset = min_offset; offset < max_offset; ++offset) {
			int event_idx = event_at_offset(band_idx, offset);
			int kmer_idx = kmer_at_offset(band_idx, offset);

			unsigned int kmer_rank = kmer_ranks[kmer_idx];
 
			int offset_up   = band_event_to_offset(band_idx - 1, event_idx - 1); 
			int offset_left = band_kmer_to_offset(band_idx - 1, kmer_idx - 1);
			int offset_diag = band_kmer_to_offset(band_idx - 2, kmer_idx - 1);

			float up   = is_offset_valid(offset_up)   ? bands[band_idx - 1][offset_up]   : -INFINITY;
			float left = is_offset_valid(offset_left) ? bands[band_idx - 1][offset_left] : -INFINITY;
			float diag = is_offset_valid(offset_diag) ? bands[band_idx - 2][offset_diag] : -INFINITY;
 
			float lp_emission = logProbabilityMatch(kmer_rank, r.events[event_idx], s.shift, s.scale);

			float score_d = diag + lp_step + lp_emission;
			float score_u = up + lp_stay + lp_emission;
			float score_l = left + lp_skip;

			float max_score = score_d;
			uint8_t from = FROM_D;

			max_score = score_u > max_score ? score_u : max_score;
			from = max_score == score_u ? FROM_U : from;
			max_score = score_l > max_score ? score_l : max_score;
			from = max_score == score_l ? FROM_L : from;
	    
			bands[band_idx][offset] = max_score;
			trace[band_idx][offset] = from;
			fills += 1;
		}
	}

	//benchmarking
	//std::chrono::steady_clock::time_point tp2 = std::chrono::steady_clock::now();
	//std::cout << "banded alignment: " << std::chrono::duration_cast<std::chrono::microseconds>(tp2 - tp1).count() << std::endl;

	//
	// Backtrack to compute alignment
	//
	double sum_emission = 0.;
	double n_aligned_events = 0;
    
	float max_score = -INFINITY;
	int curr_event_idx = 0;
	int curr_kmer_idx = n_kmers -1;

	// Find best score between an event and the last k-mer. after trimming the remaining evnets
	for(unsigned int event_idx = 0; event_idx < n_events; ++event_idx) {
		unsigned int band_idx = event_kmer_to_band(event_idx, curr_kmer_idx);
		assert(band_idx < bands.size());
		int offset = band_event_to_offset(band_idx, event_idx);
		if(is_offset_valid(offset)) {
			float s = bands[band_idx][offset] + (n_events - event_idx) * lp_trim;
			if(s > max_score) {
				max_score = s;
				curr_event_idx = event_idx;
			}
		}
	}
//end adapted from nanopolish

	//benchmarking
	//std::chrono::steady_clock::time_point tp3 = std::chrono::steady_clock::now();
	//std::cout << "calculate end index: " << std::chrono::duration_cast<std::chrono::microseconds>(tp3 - tp2).count() << std::endl;

	r.eventAlignment.reserve(r.events.size());

	int curr_gap = 0;
	int max_gap = 0;
	//int usedInScale = 0;

	while(curr_kmer_idx >= 0 && curr_event_idx >= 0) {
        
		// emit alignment
		r.eventAlignment.push_back(std::make_pair(curr_event_idx, curr_kmer_idx));

		// qc stats
		unsigned int kmer_rank = kmer_ranks[curr_kmer_idx];
		std::pair<double,double> meanStd = Pore_Substrate_Config.pore_model[kmer_rank];
		float logProbability = logProbabilityMatch(kmer_rank, r.events[curr_event_idx], s.shift, s.scale);
		sum_emission += logProbability;

		n_aligned_events += 1;
		
		//TESTING - print the alignment
		//double mu = meanStd.first;
		//double sigma = meanStd.second;
		//double x = (raw[curr_event_idx] - s.shift)/s.scale;
		//std::cout << curr_kmer_idx << "\t" << x << "\t" << mu << std::endl;
		//ENDTESTING

		int band_idx = event_kmer_to_band(curr_event_idx, curr_kmer_idx);
		int offset = band_event_to_offset(band_idx, curr_event_idx);
		assert(band_kmer_to_offset(band_idx, curr_kmer_idx) == offset);

		uint8_t from = trace[band_idx][offset];
		if(from == FROM_D) {
			curr_kmer_idx -= 1;
			curr_event_idx -= 1;
			curr_gap = 0;
		} else if(from == FROM_U) {
			curr_event_idx -= 1;
			curr_gap = 0;
		} else {
			curr_kmer_idx -= 1;
			curr_gap += 1;
			max_gap = std::max(curr_gap, max_gap);
		}   
	}
	std::reverse(r.eventAlignment.begin(), r.eventAlignment.end());

	//benchmarking
	//std::chrono::steady_clock::time_point tp4 = std::chrono::steady_clock::now();
	//std::cout << "backtrace: " << std::chrono::duration_cast<std::chrono::microseconds>(tp4 - tp3).count() << std::endl;

	// QC results
	double avg_log_emission = sum_emission / n_aligned_events;
	bool spanned = r.eventAlignment.front().second == 0 && r.eventAlignment.back().second == n_kmers - 1;
    
    	//Testing - print QCs
    	//std::cout << avg_log_emission << "\t" << spanned << "\t" << max_gap << "\t" << r.isReverse << std::endl;
    
	r.alignmentQCs.recordQCs(avg_log_emission, spanned, max_gap);
	//std::cerr << r.readID << " " << avg_log_emission << " " << max_gap << std::endl;
	if(avg_log_emission < min_average_log_emission || !spanned || max_gap > max_gap_threshold ) r.eventAlignment.clear();

	r.scalings = s;

	//benchmarking
	//std::chrono::steady_clock::time_point tp5 = std::chrono::steady_clock::now();
	//std::cout << "calculate shift and scale: " << std::chrono::duration_cast<std::chrono::microseconds>(tp5 - tp4).count() << std::endl;
}


std::vector<double> quantileMedians(std::vector<double> &data, int nquantiles){

	auto endSlice = data.end();
	
	//uncomment to downsample to a fixed number of events
	/*
	unsigned int maxEvents = 100000;
	if (data.size() > maxEvents){
		endSlice = data.begin() + maxEvents;
	}
	*/
	
	std::vector<double> data_downsample(data.begin(), endSlice);
	std::sort(data_downsample.begin(), data_downsample.end());
	
	std::vector<double> quantileMedians;
	unsigned int n = data_downsample.size() / nquantiles;
	for (int i = 0; i < nquantiles; i++){
	
		double median  = data_downsample[ (i*n + (i+1)*n)/2 ];
		quantileMedians.push_back(median);
	}
	
	return quantileMedians;
}


std::pair<double, double> linear_regression(std::vector<double> x, std::vector<double> y){

	assert(x.size() == y.size());
	
	double sum_x = 0., sum_x2 = 0., sum_y = 0., sum_xy = 0.;
	int n = y.size();
	
	for(int i = 0; i < n; i++){
		sum_x = sum_x + x[i];
		sum_x2 = sum_x2 + x[i]*x[i];
		sum_y = sum_y + y[i];
		sum_xy = sum_xy + x[i]*y[i];
	}
	
	//calculate coefficients
	double slope = (n * sum_xy - sum_x * sum_y)/(n * sum_x2 - sum_x * sum_x);
	double intercept = (sum_y - slope * sum_x)/n;
	
	//testing
	/*
	for(int i = 0; i < n; i++){
		std::cerr << x[i] << " " << y[i] << std::endl;
	}
	std::cerr << slope << " " << intercept << std::endl;
	std::cerr << "----------------------" << std::endl;	
	*/

	return std::make_pair(slope,intercept);
}


PoreParameters estimateScaling_quantiles(std::vector< double > &signal_means, std::string &sequence, std::vector<unsigned int> &kmer_ranks ){

	PoreParameters s;

	size_t k = Pore_Substrate_Config.kmer_len;
	unsigned int numOfKmers = sequence.size() - k + 1;

	std::vector<double> model_means;
	model_means.reserve(numOfKmers);
	for ( unsigned int i = 0; i < numOfKmers; i ++ ){

		std::pair<double,double> meanStd = Pore_Substrate_Config.pore_model[kmer_ranks[i]];
		double kmer_mean = meanStd.first;
		model_means.push_back(kmer_mean);
	}

	std::vector<double> signal_quantiles = quantileMedians(signal_means, 10);
	std::vector<double> model_quantiles = quantileMedians(model_means, 10);

	std::pair<double, double> scalings = linear_regression(model_quantiles, signal_quantiles);
	
	s.shift = scalings.second;
	s.scale = scalings.first;
	
	return s;
}


void normaliseEvents( read &r ){

	try{

		fast5_getSignal(r);
	}
	catch ( BadFast5Field &bf5 ){

		return;
	}
	
	
	event_table et = detect_events(&(r.raw)[0], (r.raw).size(), event_detection_defaults);
	assert(et.n > 0);
	
	r.events.reserve(et.n);
	unsigned int rawStart = 0;
	double mean = 0.;
	for ( unsigned int i = 0; i < et.n; i++ ){

		if (et.event[i].mean > 0.) {

			if (i > 0){
			
				//TESTING
				/*
				std::cout << mean << std::endl;
				for (unsigned int j = rawStart; j <= et.event[i].start-1; j++){
				
					std::cout << "\t" << r.raw[j] << std::endl;
				}
				*/
				
				
				//build the previous event
				event e;
				e.mean = mean;
				for (unsigned int j = rawStart; j <= et.event[i].start-1; j++){
				
					e.raw.push_back(r.raw[j]);
				}
				r.events.push_back(e);				
				
				//save stats for the next event
				mean = et.event[i].mean;
				rawStart = et.event[i].start;
			}
		}
	}
	
	
	// Precompute k-mer ranks for rescaling and banded alignment - query sequence
	size_t k = Pore_Substrate_Config.kmer_len;
	size_t n_kmers = r.basecall.size() - k + 1;
	std::vector<unsigned int> kmer_ranks_query(n_kmers);
	for(size_t i = 0; i < n_kmers; i++) {
		std::string kmer = r.basecall.substr(i, k);
		kmer_ranks_query[i] = kmer2index(kmer, k);
	}
	
	// Precompute k-mer ranks for rescaling and banded alignment - reference sequence
	n_kmers = r.referenceSeqMappedTo.size() - k + 1;
	std::vector<unsigned int> kmer_ranks_ref(n_kmers);
	for(size_t i = 0; i < n_kmers; i++) {
		std::string kmer = r.referenceSeqMappedTo.substr(i, k);
		kmer_ranks_ref[i] = kmer2index(kmer, k);
	}
	
	//normalise by quantile scaling by comparing the raw signal against the reference sequence
	PoreParameters s = estimateScaling_quantiles( r.raw, r.referenceSeqMappedTo, kmer_ranks_ref );
		
	// Rough alignment of signals to query sequence
	adaptive_banded_simple_event_align(r, s, kmer_ranks_query);
	
	r.scalings.eventsPerBase = (double) et.n / (double) (r.basecall.size() - k);
}
