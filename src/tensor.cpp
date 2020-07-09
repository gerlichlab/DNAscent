//----------------------------------------------------------
// Copyright 2020 University of Cambridge
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#include "tensor.h"
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


std::shared_ptr<ModelSession> model_load(const char *filename, const char *input_name, const char *output_name){

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
    uint8_t intra_op_parallelism_threads = 0;
    uint8_t inter_op_parallelism_threads = 0;
    uint8_t buf[]={0x10,intra_op_parallelism_threads,0x28,inter_op_parallelism_threads};
    TF_SetConfig(opts, buf,sizeof(buf),status.ptr);
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
