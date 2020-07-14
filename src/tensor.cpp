//----------------------------------------------------------
// Copyright 2020 University of Cambridge
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#include "tensor.h"
//#include "../tensorflow/include/tensorflow/c/c_api.h"
#include <fstream>


static TF_Buffer* read_tf_buffer_from_file(const char* file) {
	std::ifstream t(file, std::ifstream::binary);
	t.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	t.seekg(0, std::ios::end);
	size_t size = t.tellg();
	auto data = std::make_unique<char[]>(size);
	t.seekg(0);
	t.read(data.get(), size);

	TF_Buffer *buf = TF_NewBuffer();
	buf->data = data.release();
	buf->length = size;
	buf->data_deallocator = free_cpp_array<char>;
	return buf;
}


std::shared_ptr<ModelSession> model_load_cpu(const char *filename, const char *input_name, const char *output_name){

	int vis = setenv("CUDA_VISIBLE_DEVICES", "", 1);
	if (vis == -1){
		std::cerr << "Suppression of GPU devices failed." << std::endl;
	}

	CStatus status;
	std::shared_ptr<ModelSession> ms = std::make_shared<ModelSession>();
	std::shared_ptr<TF_Graph *> graph = std::make_shared<TF_Graph *>(TF_NewGraph());

	{
        // Load a protobuf containing a GraphDef
        auto graph_def=read_tf_buffer_from_file(filename);
        if(!graph_def) return nullptr;
        auto graph_opts=TF_NewImportGraphDefOptions();
        TF_GraphImportGraphDef(*(graph.get()), graph_def, graph_opts, status.ptr);
	}

	if(status.failure()){
		status.dump_error();
		return nullptr;
	}
	ms -> graph = graph;

	auto input_op = TF_GraphOperationByName(*(graph.get()), input_name);
	auto output_op = TF_GraphOperationByName(*(graph.get()), output_name);
	if(!input_op || !output_op){
		return nullptr;
	}

	TF_SessionOptions *opts = TF_NewSessionOptions();

	//uncomment to cap CPU threads
    //uint8_t intra_op_parallelism_threads = 1;
    //uint8_t inter_op_parallelism_threads = 1;
    //uint8_t buf[]={0x10,intra_op_parallelism_threads,0x28,inter_op_parallelism_threads};
    //TF_SetConfig(opts, buf,sizeof(buf),status.ptr);

	std::shared_ptr<TF_Session*> session = std::make_shared<TF_Session*>(TF_NewSession(*(graph.get()), opts, status.ptr));

	if(status.failure()){
		return nullptr;
	}
	assert(session);
	ms -> session = session;

	ms -> inputs = {input_op, 0};
	ms -> outputs = {output_op, 0};

	return ms;
}


std::shared_ptr<ModelSession> model_load_gpu(const char *filename, const char *input_name, const char *output_name, unsigned char device){

	CStatus status;
	std::shared_ptr<ModelSession> ms = std::make_shared<ModelSession>();
	std::shared_ptr<TF_Graph *> graph = std::make_shared<TF_Graph *>(TF_NewGraph());

	{
        // Load a protobuf containing a GraphDef
        auto graph_def=read_tf_buffer_from_file(filename);
        if(!graph_def) return nullptr;
        auto graph_opts=TF_NewImportGraphDefOptions();
        TF_GraphImportGraphDef(*(graph.get()), graph_def, graph_opts, status.ptr);
	}

	if(status.failure()){
		status.dump_error();
		return nullptr;
	}
	ms -> graph = graph;

	auto input_op = TF_GraphOperationByName(*(graph.get()), input_name);
	auto output_op = TF_GraphOperationByName(*(graph.get()), output_name);
	if(!input_op || !output_op){
		return nullptr;
	}

	//the buffer that follows is equivalent to:
	//config = tf.ConfigProto(allow_soft_placement=True,device_count = {'GPU': 1,'CPU':1},intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
	//config.gpu_options.allow_growth=True
	//config.gpu_options.per_process_gpu_memory_fraction = 0.95
	//config.gpu_options.visible_device_list= <device_name>

	TF_SessionOptions *opts = TF_NewSessionOptions();
    uint8_t buf[]={0xa, 0x7, 0xa, 0x3, 0x47, 0x50, 0x55, 0x10, 0x1, 0xa, 0x7, 0xa, 0x3, 0x43, 0x50, 0x55, 0x10, 0x1, 0x10, 0x1, 0x28, 0x1, 0x32, 0xe, 0x9, 0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xee, 0x3f, 0x20, 0x1, 0x2a, 0x1, device, 0x38, 0x1};
    TF_SetConfig(opts, buf,sizeof(buf),status.ptr);
    //TF_EnableXLACompilation(opts,true);
	std::shared_ptr<TF_Session*> session = std::make_shared<TF_Session*>(TF_NewSession(*(graph.get()), opts, status.ptr));

	if(status.failure()){
		return nullptr;
	}
	assert(session);
	ms -> session = session;

	ms -> inputs = {input_op, 0};
	ms -> outputs = {output_op, 0};

	return ms;
}

