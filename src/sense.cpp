//----------------------------------------------------------
// Copyright 2019 University of Oxford
// Written by Michael A. Boemo (mb915@cam.ac.uk)
// This software is licensed under GPL-3.0.  You should have
// received a copy of the license with this software.  If
// not, please Email the author.
//----------------------------------------------------------

#include "../tensorflow/include/tensorflow/c/c_api.h"
#include <fstream>
#include "regions.h"
#include "data_IO.h"
#include "train.h"
#include "sense.h"
#include <cmath>
#include <memory>
#include <math.h>
#include <algorithm>
#include <limits>

static const char *help=
"forkSense: DNAscent AI executable that calls replication origins, fork movement, and fork stalling.\n"
"To run DNAscent forkSense, do:\n"
"  ./DNAscent forkSense [arguments]\n"
"Example:\n"
"  ./DNAscent forkSense -d /path/to/BrdUCalls.detect -o /path/to/output.forkSense\n"
"Required arguments are:\n"
"  -d,--detect               path to output file from DNAscent detect,\n"
"  -o,--output               path to output file for forkSense.\n"
"Optional arguments are:\n"
"  -t,--threads              number of threads (default: 1 thread),\n"
"  -l,--likelihood           log-likelihood threshold for a positive analogue call (default: 1.25),\n"
"  -c,--cooldown             minimum gap between positive analogue calls (default: 4),\n"
"  --markOrigins             writes replication origin locations to a bed file (default: off),\n"
"  --markStalls              writes fork stall locations to a bed file (default: off).\n"
"Written by Michael Boemo, Department of Pathology, University of Cambridge.\n"
"Please submit bug reports to GitHub Issues (https://github.com/MBoemo/DNAscent/issues).";

struct Arguments {

	std::string detectFilename;
	std::string outputFilename;
	bool markOrigins = false;
	bool markStalls = false;
	unsigned int threads = 1;
	int cooldown;
	double likelihood;

};

Arguments parseSenseArguments( int argc, char** argv ){

 	if( argc < 2 ){
 		std::cout << "Exiting with error.  Insufficient arguments passed to DNAscent regions." << std::endl << help << std::endl;
		exit(EXIT_FAILURE);
	}
 	if ( std::string( argv[ 1 ] ) == "-h" or std::string( argv[ 1 ] ) == "--help" ){
 		std::cout << help << std::endl;
		exit(EXIT_SUCCESS);
	}
	else if( argc < 4 ){
 		std::cout << "Exiting with error.  Insufficient arguments passed to DNAscent regions." << std::endl;
		exit(EXIT_FAILURE);
	}

 	Arguments args;
	args.likelihood = 1.25;
	args.cooldown = 4;

 	/*parse the command line arguments */
	for ( int i = 1; i < argc; ){

 		std::string flag( argv[ i ] );

 		if ( flag == "-d" or flag == "--detect" ){
 			std::string strArg( argv[ i + 1 ] );
			args.detectFilename = strArg;
			i+=2;
		}
		else if ( flag == "-o" or flag == "--output" ){
 			std::string strArg( argv[ i + 1 ] );
			args.outputFilename = strArg;
			i+=2;
		}
		else if ( flag == "-t" or flag == "--threads" ){

			std::string strArg( argv[ i + 1 ] );
			args.threads = std::stoi( strArg.c_str() );
			i+=2;
		}
		else if ( flag == "--markOrigins" ){

			args.markOrigins = true;
			i+=1;
		}
		else if ( flag == "--markStalls" ){

			args.markStalls = true;
			i+=1;
		}
		else if ( flag == "-c" or flag == "--cooldown" ){
 			std::string strArg( argv[ i + 1 ] );
			args.cooldown = std::stoi( strArg.c_str() );
			i+=2;
		}
		else if ( flag == "-l" or flag == "--likelihood" ){
 			std::string strArg( argv[ i + 1 ] );
			args.likelihood = std::stof( strArg.c_str() );
			i+=2;
		}
		else throw InvalidOption( flag );
	}
	if (args.outputFilename == args.detectFilename) throw OverwriteFailure();

	return args;
}


static TF_Buffer *read_tf_buffer_from_file(const char* file);


std::vector<int> pooling = {6,4,4,4};


class CStatus{
	public:
		TF_Status *ptr;
		CStatus(){
			ptr = TF_NewStatus();
		}

		void dump_error()const{
			std::cerr << "TF status error: " << TF_Message(ptr) << std::endl;
		}

		inline bool failure()const{
			return TF_GetCode(ptr) != TF_OK;
		}

		~CStatus(){
			if(ptr)TF_DeleteStatus(ptr);
		}
};

namespace detail {
	template<class T>
	class TFObjDeallocator;

	template<>
	struct TFObjDeallocator<TF_Status> { static void run(TF_Status *obj) { TF_DeleteStatus(obj); }};

	template<>
	struct TFObjDeallocator<TF_Graph> { static void run(TF_Graph *obj) { TF_DeleteGraph(obj); }};

	    template<>
	struct TFObjDeallocator<TF_Tensor> { static void run(TF_Tensor *obj) { TF_DeleteTensor(obj); }};

	template<>
	struct TFObjDeallocator<TF_SessionOptions> { static void run(TF_SessionOptions *obj) { TF_DeleteSessionOptions(obj); }};

	template<>
	struct TFObjDeallocator<TF_Buffer> { static void run(TF_Buffer *obj) { TF_DeleteBuffer(obj); }};

	template<>
	struct TFObjDeallocator<TF_ImportGraphDefOptions> {
		static void run(TF_ImportGraphDefOptions *obj) { TF_DeleteImportGraphDefOptions(obj); }
	};

	template<>
	struct TFObjDeallocator<TF_Session> {
		static void run(TF_Session *obj) {
			CStatus status;
			TF_DeleteSession(obj, status.ptr);
			if (status.failure()) {
				status.dump_error();
			}
		}
	};
}

template<class T> struct TFObjDeleter{
	void operator()(T* ptr) const{
		detail::TFObjDeallocator<T>::run(ptr);
	}
};

template<class T> struct TFObjMeta{
	typedef std::unique_ptr<T, TFObjDeleter<T>> UniquePtr;
};

template<class T> typename TFObjMeta<T>::UniquePtr tf_obj_unique_ptr(T *obj){
	typename TFObjMeta<T>::UniquePtr ptr(obj);
	return ptr;
}

class MySession{
	public:
		typename TFObjMeta<TF_Graph>::UniquePtr graph;
		typename TFObjMeta<TF_Session>::UniquePtr session;

		TF_Output inputs, outputs;
};


MySession *my_model_load(const char *filename, const char *input_name, const char *output_name){
	//printf("Loading model %s\n", filename);
	CStatus status;

	auto graph=tf_obj_unique_ptr(TF_NewGraph());
	{
        // Load a protobuf containing a GraphDef
        auto graph_def=tf_obj_unique_ptr(read_tf_buffer_from_file(filename));
	if(!graph_def){
		return nullptr;
        }

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

	auto session = std::make_unique<MySession>();
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


template<class T> static void free_cpp_array(void* data, size_t length){
	delete []((T *)data);
}


template<class T> static void cpp_array_deallocator(void* data, size_t length, void* arg){
	delete []((T *)data);
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

#define MY_TENSOR_SHAPE_MAX_DIM 16
struct TensorShape{
	int64_t values[MY_TENSOR_SHAPE_MAX_DIM];
	int dim;

	int64_t size()const{
		assert(dim>=0);
		int64_t v=1;
		for(int i=0;i<dim;i++)v*=values[i];
		return v;
    }
};


TF_Tensor *read2tensor(DetectedRead &r, const TensorShape &shape){

	size_t size = r.brduCalls.size() * 4;
	//put a check in here for size

	r.generateInput();

	auto output_array = std::make_unique<float[]>(size);
	{
		for(size_t i = 0; i < size; i++){
			output_array[i] = r.tensorInput[i];
		}
	}

	auto output = tf_obj_unique_ptr(TF_NewTensor(TF_FLOAT,
		shape.values, shape.dim,
		(void *)output_array.get(), size*sizeof(float), cpp_array_deallocator<float>, nullptr));
	if(output){
		// The ownership has been successfully transferred
		output_array.release();
	}
	return output.release();
}


std::string callStalls(DetectedRead &r){

	assert(r.positions.size() == r.probabilities.size());

	float threshold = 0.5;
	bool inStall = false;
	int stallStart = -1, potentialEnd = -1;
	std::string outBed;

	for (size_t i = 1; i < r.probabilities.size(); i++){

		if (r.probabilities[i][3] > threshold and not inStall){ //initilise the stall site

			stallStart = r.positions[i];
			inStall = true;
		}
		else if (inStall and r.probabilities[i][3] < threshold and r.probabilities[i-1][3] > threshold){

			potentialEnd = r.positions[i];
		}
		else if (inStall and (r.probabilities[i][0] > threshold or r.probabilities[i][1] > threshold or r.probabilities[i][2] > threshold)){//close if we've confidently moved to something else

			assert(stallStart != -1 and potentialEnd != -1);
			r.stalls.push_back(std::make_pair(stallStart,potentialEnd));
			outBed += r.chromosome + " " + std::to_string(stallStart) + " " + std::to_string(potentialEnd) + " " + r.header.substr(1) + "\n";
			inStall = false;
			stallStart = -1;
			potentialEnd = -1;

		}
	}

	//if we got to the end of the read without closing
	if (inStall){

		assert(stallStart != -1);
		if (potentialEnd == -1) potentialEnd = r.positions.back();

		r.stalls.push_back(std::make_pair(stallStart,potentialEnd));
		outBed += r.chromosome + " " + std::to_string(stallStart) + " " + std::to_string(potentialEnd) + " " + r.header.substr(1) + "\n";
	}

	return outBed;
}


std::string callOrigins(DetectedRead &r, bool stallsMarked){

	//detect stalls if we haven't already
	//if (not stallsMarked) callStalls(r);

	assert(r.positions.size() == r.probabilities.size());

	float threshold = 0.8;

	std::vector<std::pair<int,int>> leftForks, rightForks;
	std::string outBed;

	bool inFork = false;
	int forkStart = -1, potentialEnd = -1;

	//rightward-moving forks
	for (size_t i = 1; i < r.probabilities.size(); i++){

		if (r.probabilities[i][2] > threshold and not inFork){ //initialise the site

			forkStart = r.positions[i];
			inFork = true;
		}
		else if (inFork and r.probabilities[i][2] < threshold and r.probabilities[i-1][2] > threshold){

			potentialEnd = r.positions[i];
		}
		else if (inFork and (r.probabilities[i][0] > threshold or r.probabilities[i][1] > threshold)){//close if we've confidently moved to something else

			assert(forkStart != -1 and potentialEnd != -1);
			rightForks.push_back(std::make_pair(forkStart,potentialEnd));
			inFork = false;
			forkStart = -1;
			potentialEnd = -1;

		}
	}

	//if we got to the end of the read without closing
	if (inFork){

		assert(forkStart != -1);
		if (potentialEnd == -1) potentialEnd = r.positions.back();

		rightForks.push_back(std::make_pair(forkStart,potentialEnd));
	}

	inFork = false;
	forkStart = -1;
	potentialEnd = -1;

	//reverse order for leftward-moving forks
	std::vector<int> revPositions(r.positions.rbegin(), r.positions.rend());
	std::vector<std::vector<float>> revProbabilities(r.probabilities.rbegin(), r.probabilities.rend());

	//leftward-moving forks
	for (size_t i = 1; i < revProbabilities.size(); i++){

		if (revProbabilities[i][0] > threshold and not inFork){ //initialise the site

			forkStart = revPositions[i];
			inFork = true;
		}
		else if (inFork and revProbabilities[i][0] < threshold and revProbabilities[i-1][0] > threshold){

			potentialEnd = revPositions[i];
		}
		else if (inFork and (revProbabilities[i][1] > threshold or revProbabilities[i][2] > threshold)){//close if we've confidently moved to something else

			assert(forkStart != -1 and potentialEnd != -1);
			leftForks.push_back(std::make_pair(potentialEnd,forkStart));
			inFork = false;
			forkStart = -1;
			potentialEnd = -1;

		}
	}

	//if we got to the end of the read without closing
	if (inFork){

		assert(forkStart != -1);
		if (potentialEnd == -1) potentialEnd = revPositions.back();

		leftForks.push_back(std::make_pair(potentialEnd,forkStart));
	}

	//match up regions
	for ( size_t li = 0; li < leftForks.size(); li++ ){

		//find the closest right fork region
		int minDist = std::numeric_limits<int>::max();
		int bestMatch = -1;
		for ( size_t ri = 0; ri < rightForks.size(); ri++ ){

			if (leftForks[li].second > rightForks[ri].first ) continue;

			int dist = rightForks[ri].first - leftForks[li].second;
			if (dist < minDist){
				minDist = dist;
				bestMatch = ri;

			}
		}

		//make sure no other left forks are closer
		bool failed = false;
		if (bestMatch != -1){

			for (size_t l2 = li+1; l2 < leftForks.size(); l2++){

				if (leftForks[l2].second > rightForks[bestMatch].first ) continue;

				int dist = rightForks[bestMatch].first - leftForks[l2].second;
				if (dist < minDist){

					failed = true;
					break;
				}
			}
		}
		if (failed) continue;
		else if (bestMatch != -1){

			r.origins.push_back(std::make_pair(leftForks[li].second, rightForks[bestMatch].first));
			outBed += r.chromosome + " " + std::to_string(leftForks[li].second) + " " + std::to_string(rightForks[bestMatch].first) + " " + r.header.substr(1) + "\n";
		}
	}

	return outBed;
}


std::string runCNN(DetectedRead &r, std::string modelPath){

	auto session = std::unique_ptr<MySession>(my_model_load(modelPath.c_str(), "conv1d_input", "time_distributed_2/Reshape_1"));
	TensorShape input_shape={{1, r.brduCalls.size(), 4}, 3};
	auto input_values = tf_obj_unique_ptr(read2tensor(r, input_shape));
	if(!input_values){
		std::cerr << "Tensor creation failure." << std::endl;
		exit (EXIT_FAILURE);
	}

	CStatus status;
	TF_Tensor* inputs[]={input_values.get()};
	TF_Tensor* outputs[1]={};
	TF_SessionRun(session->session.get(), nullptr,
		&session->inputs, inputs, 1,
		&session->outputs, outputs, 1,
		nullptr, 0, nullptr, status.ptr);
	auto _output_holder = tf_obj_unique_ptr(outputs[0]);

	if(status.failure()){
		status.dump_error();
		exit (EXIT_FAILURE);
	}

	TF_Tensor &output = *outputs[0];
	if(TF_TensorType(&output) != TF_FLOAT){
		std::cerr << "Error, unexpected output tensor type." << std::endl;
		exit (EXIT_FAILURE);
	}

	std::string str_output;
	unsigned int outputFields = 3;
	{
		str_output += r.readID + " " + r.chromosome + " " + std::to_string(r.mappingLower) + " " + std::to_string(r.mappingUpper) + " " + r.strand + "\n"; //header
		size_t output_size = TF_TensorByteSize(&output) / sizeof(float);
		assert(output_size == r.brduCalls.size() * outputFields);
		auto output_array = (const float *)TF_TensorData(&output);
		unsigned int pos = 0;
		str_output += std::to_string(r.positions[0]);
		for(size_t i = 0; i < output_size; i++){
			str_output += "\t" + std::to_string(output_array[i]);
			if((i+1)%outputFields==0){
				r.probabilities.push_back({output_array[i-2],output_array[i-1],output_array[i]});
				str_output += "\n";
				pos++;
				if (i != output_size-1) str_output += std::to_string(r.positions[pos]);

			}
		}
	}
	return str_output;
}


void emptyBuffer(std::vector< DetectedRead > &buffer, Arguments args, std::string modelPath, std::ofstream &outFile, std::ofstream &originFile, std::ofstream &stallFile, int trimFactor){

	#pragma omp parallel for schedule(dynamic) shared(args,outFile) num_threads(args.threads)
	for ( auto b = buffer.begin(); b < buffer.end(); b++) {

		b -> trim(trimFactor);
		std::string readOutput = runCNN(*b, modelPath);

		std::string stallOutput, originOutput;
		if (args.markStalls){

			stallOutput = callStalls(*b);
		}
		if (args.markOrigins){

			originOutput = callOrigins(*b,args.markStalls);
		}

		#pragma omp critical
		{
			outFile << readOutput;
			if (args.markStalls and (*b).stalls.size() > 0) stallFile << stallOutput;
			if (args.markOrigins and (*b).origins.size() > 0) originFile << originOutput;
		}
	}
	buffer.clear();
}


bool checkReadLength( int length ){

	for (auto p = pooling.begin(); p < pooling.end(); p++){

		length /= *p;
	}
	if (length <= 3) return false;
	else return true;
}


int sense_main( int argc, char** argv ){

	Arguments args = parseSenseArguments( argc, argv );

	unsigned int maxBufferSize = 20*(args.threads);

	//get the model
	std::string pathExe = getExePath();
	std::string modelPath = pathExe + "/dnn_models/" + "forks.pb";

	//get a read count
	int readCount = 0;
	std::string line;
	std::ifstream inFile( args.detectFilename );
	if ( not inFile.is_open() ) throw IOerror( args.detectFilename );
	while( std::getline( inFile, line ) ){

		if ( line.substr(0,1) == ">" ) readCount++;
	}	
	progressBar pb(readCount,false);
	inFile.close();

	//open all the files
 	inFile.open( args.detectFilename );
	if ( not inFile.is_open() ) throw IOerror( args.detectFilename );
 	std::ofstream outFile( args.outputFilename );
	if ( not outFile.is_open() ) throw IOerror( args.outputFilename );
 	std::ofstream originFile, stallFile;
	if (args.markStalls){

		stallFile.open("forkStalls_DNAscent_forkSense.bed");
		if ( not stallFile.is_open() ) throw IOerror( "forkStalls_DNAscent_forkSense.bed" );
	}
	if (args.markOrigins){

		originFile.open("origins_DNAscent_forkSense.bed");
		if ( not originFile.is_open() ) throw IOerror( "origins_DNAscent_forkSense.bed" );
	}

	//compute trim factor
	unsigned int trimFactor = 1;
	for (auto p = pooling.begin(); p < pooling.end(); p++) trimFactor *= *p;

	std::vector< DetectedRead > readBuffer;
	int progress = 0;
	int callCooldown = 0;
	int attemptCooldown = 0;
	while( std::getline( inFile, line ) ){

		if ( line.substr(0,1) == ">" ){

			//check the read length on the back of the buffer
			if (readBuffer.size() > 0){

				bool longEnough = checkReadLength( readBuffer.back().positions.size() );
				if (not longEnough) readBuffer.pop_back();
			}

			//empty the buffer if it's full
			if (readBuffer.size() >= maxBufferSize) emptyBuffer(readBuffer, args, modelPath, outFile, originFile, stallFile, trimFactor);

			progress++;
			pb.displayProgress( progress, 0, 0 );

			DetectedRead d;
			d.header = line;
			std::stringstream ssLine(line);
			std::string column;
			int cIndex = 0;
			while ( std::getline( ssLine, column, ' ' ) ){

				if ( cIndex == 0 ) d.readID = column;
				else if ( cIndex == 1 ) d.chromosome = column;
				else if ( cIndex == 2 ) d.mappingLower = std::stoi(column);
				else if ( cIndex == 3 ) d.mappingUpper = std::stoi(column);
				else if ( cIndex == 4 ) d.strand = column;
				else throw DetectParsing();
				cIndex++;
			}
			assert(d.mappingUpper > d.mappingLower);
			readBuffer.push_back(d);
			callCooldown = 0;
			attemptCooldown = 0;
		}
		else{

			std::string column;
			std::stringstream ssLine(line);
			int position = -1, cIndex = 0;
			AnalogueScore B, BM;
			int countCol = std::count(line.begin(), line.end(), '\t');
			while ( std::getline( ssLine, column, '\t' ) ){

				if ( cIndex == 0 ){

					position = std::stoi(column);
				}
				else if ( cIndex == 1 ){

					B.set(std::stof(column));
				}
				else if ( cIndex == 2 and countCol > 3 ){ //methyl-aware detect file

					BM.set(std::stof(column));
				}
				cIndex++;
			}

			if ( B.get() > args.likelihood and position - callCooldown >= args.cooldown ){

				readBuffer.back().positions.push_back(position);
				readBuffer.back().brduCalls.push_back(B.get());
				attemptCooldown = position;
				callCooldown = position;
			}
			else if ( B.get() < args.likelihood and position - attemptCooldown >= args.cooldown) {

				readBuffer.back().positions.push_back(position);
				readBuffer.back().brduCalls.push_back(B.get());
				attemptCooldown = position;
			}
		}
	}

	//empty the buffer at the end
	emptyBuffer(readBuffer, args, modelPath, outFile, originFile, stallFile, trimFactor);

	inFile.close();
	outFile.close();
	if (args.markStalls) stallFile.close();
	if (args.markOrigins) originFile.close();

	std::cout << std::endl << "Done." << std::endl;

	return 0;
}

