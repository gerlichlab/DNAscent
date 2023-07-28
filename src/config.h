#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include "data_IO.h"

struct HMM_TransitionProbs {
	double externalD2D;
	double externalD2M;
	double externalI2M;
	double externalM2D;

	double internalM2I;
	double internalI2I;
};


struct AdaptiveBanded_Params {
	double min_average_log_emission;
	int max_gap_threshold;
	int bandwidth;
};


class Global_Config{

	public:
		unsigned int kmer_len, windowLength_align;
		std::string fn_pore_model, fn_analogue_model, fn_dnn_model, dnn_model_inputLayer1, dnn_model_inputLayer2;
		std::vector< std::pair< double, double > > pore_model, analogue_model;
		HMM_TransitionProbs HMM_config;
		AdaptiveBanded_Params AdaptiveBanded_config;

		AdaptiveBanded_Params AdaptiveBanded_Params_DNA_R10{-0.15, 5, 100}; //DNA - R10.4.1

		HMM_TransitionProbs HMM_TransitionProbs_DNA_R10{0.3, 0.7, 0.999, 0.0025, 0.001, 0.001}; //DNA - R10.4.1

		void configure_DNA_R10(void){
			kmer_len = 9;
			windowLength_align = 50;
			
			fn_pore_model = "r10.4.1_unlabelled_cauchy.model";
			fn_analogue_model = "r10.4.1_BrdU_cauchy.model";
			pore_model = import_poreModel(fn_pore_model, kmer_len);
			analogue_model = import_poreModel(fn_analogue_model, kmer_len);
			
			fn_dnn_model = "dnn_models/detect_model_BrdUEdU_DNAr10_4_1/";
			dnn_model_inputLayer1 = "serving_default_input_1";
			dnn_model_inputLayer2 = "serving_default_input_2";
			
			HMM_config = HMM_TransitionProbs_DNA_R10;
			AdaptiveBanded_config = AdaptiveBanded_Params_DNA_R10;
		}
};

extern Global_Config Pore_Substrate_Config;

#endif
