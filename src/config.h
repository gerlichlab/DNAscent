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
		unsigned int kmer_len;
		std::string fn_pore_model, fn_analogue_model;
		std::vector< std::pair< double, double > > pore_model, analogue_model;
		HMM_TransitionProbs HMM_config;
		AdaptiveBanded_Params AdaptiveBanded_config;

		AdaptiveBanded_Params AdaptiveBanded_Params_DNA_R9{-5.0, 50, 100}; //DNA - R9.4.1
		AdaptiveBanded_Params AdaptiveBanded_Params_RNA_R9{-5.0, 50, 100}; //RNA - R9.4.1

		HMM_TransitionProbs HMM_TransitionProbs_DNA_R9{0.3, 0.7, 0.999, 0.0025, 0.001, 0.001}; //DNA - R9.4.1
		HMM_TransitionProbs HMM_TransitionProbs_RNA_R9{0.3, 0.7, 0.999, 0.0025, 0.001, 0.001}; //RNA - R9.4.1

		void configure_DNA_R9(void){
			kmer_len = 6;
			fn_pore_model = "DNA_R9.4.1._template_median68pA.6mer.model";
			fn_analogue_model = "DNA_R9.4.1_BrdU.model";
			pore_model = import_poreModel(fn_pore_model, kmer_len);
			analogue_model = import_poreModel(fn_analogue_model, kmer_len);

			HMM_config = HMM_TransitionProbs_DNA_R9;
			AdaptiveBanded_config = AdaptiveBanded_Params_DNA_R9;
		}
		void configure_RNA_R9(void){
			kmer_len = 5;
			fn_pore_model = "RNA_R9.4.1_180mv_70bps_5mer_RNA.model";
			pore_model = import_poreModel(fn_pore_model, kmer_len);
			HMM_config = HMM_TransitionProbs_RNA_R9;
			AdaptiveBanded_config = AdaptiveBanded_Params_RNA_R9;
		}
};

extern Global_Config Pore_Substrate_Config;

#endif
