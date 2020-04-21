//----------------------------------------------------------
// Copyright 2020 University of Cambridge
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#include "tensor.h"
#include <fstream>


ModelSession *model_load(const char *filename, const char *input_name, const char *output_name){
	//printf("Loading model %s\n", filename);
	CStatus status;

	auto graph=tf_obj_unique_ptr(TF_NewGraph());
	{
        // Load a protobuf containing a GraphDef
        auto graph_def=tf_obj_unique_ptr(read_tf_buffer_from_file(filename));
        if(!graph_def) return nullptr;

        auto graph_opts=tf_obj_unique_ptr(TF_NewImportGraphDefOptions());
        TF_GraphImportGraphDef(graph.get(), graph_def.get(), graph_opts.get(), status.ptr);
	}

	if(status.failure()){
		status.dump_error();
		return nullptr;
	}

	auto input_op = TF_GraphOperationByName(graph.get(), input_name);
	auto output_op = TF_GraphOperationByName(graph.get(), output_name);
	if(!input_op || !output_op){
		return nullptr;
	}

	auto session = std::make_unique<ModelSession>();
	{
		auto opts = tf_obj_unique_ptr(TF_NewSessionOptions());
		session->session = tf_obj_unique_ptr(TF_NewSession(graph.get(), opts.get(), status.ptr));
	}

	if(status.failure()){
		return nullptr;
	}
	assert(session);

	graph.swap(session->graph);
	session->inputs = {input_op, 0};
	session->outputs = {output_op, 0};

	return session.release();
}


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


