#include <assert.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>

#include <TH/TH.h>
#include "NvInfer.h"
#include "NvCaffeParser.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

#define CHECK(status)									\
{														\
	if (status != 0)									\
	{													\
		std::cout << "Cuda failure: " << status;		\
		abort();										\
	}													\
}

extern "C" 
{
void init(void* handle[1],
		  const char* deployFile,
		  const char* modelFile,				 
		  unsigned int maxBatchSize);

void caffeToGIEModel_(const char* deployFile,				
					  const char* modelFile,				
					  unsigned int maxBatchSize,					
					  std::ostream& gieModelStream);	

void doInference(void* handle[1], THFloatTensor* input, THFloatTensor* output_mask, THFloatTensor* output_box, int batchSize);

}

// Logger for GIE info/warning/errors
class Logger : public ILogger			
{
	void log(Severity severity, const char* msg) override
	{
		// suppress info-level messages
		if (severity != Severity::kINFO)
			std::cout << msg << std::endl;
	}
} gLogger;

void init(void* handle[1],
		  const char* deployFile,
		  const char* modelFile,				 
		  unsigned int maxBatchSize)	   
{
	// create a GIE model from the caffe model and serialize it to a stream
	std::stringstream gieModelStream;
	caffeToGIEModel_(deployFile, modelFile, maxBatchSize, gieModelStream);

	// deserialize the engine 
	gieModelStream.seekg(0, gieModelStream.beg);

	IRuntime* runtime = createInferRuntime(gLogger);
	ICudaEngine* engine = runtime->deserializeCudaEngine(gieModelStream);
	IExecutionContext *context = engine->createExecutionContext();

	handle[1] = context;
}

void caffeToGIEModel_(const char* deployFile,				
					  const char* modelFile,				
					  unsigned int maxBatchSize,					
					  std::ostream& gieModelStream)				
{
	// create the builder
	IBuilder* builder = createInferBuilder(gLogger);

	// parse the caffe model to populate the network, then set the outputs
	INetworkDefinition* network = builder->createNetwork();
	ICaffeParser* parser = createCaffeParser();
	const IBlobNameToTensor* blobNameToTensor = parser->parse(deployFile,
															  modelFile,
															  *network,
															  DataType::kFLOAT);

	// specify which tensors are outputs
	network->markOutput(*blobNameToTensor->find("coverage"));
	network->markOutput(*blobNameToTensor->find("bboxes"));

	// Build the engine
	builder->setMaxBatchSize(maxBatchSize);
	builder->setMaxWorkspaceSize(16 << 20);

	ICudaEngine* engine = builder->buildCudaEngine(*network);
	assert(engine);

	// we don't need the network any more, and we can destroy the parser
	network->destroy();
	parser->destroy();

	// serialize the engine, then close everything down
	engine->serialize(gieModelStream);
	engine->destroy();
	builder->destroy();
	shutdownProtobufLibrary();
}

void doInference(void* handle[1], THFloatTensor* input, THFloatTensor* output_mask, THFloatTensor* output_box, int batchSize)
{
	int im_h = THFloatTensor_size(input, 1);
	int im_w = THFloatTensor_size(input, 2);
	int mask_h = THFloatTensor_size(output_mask, 1);
	int mask_w = THFloatTensor_size(output_mask, 2);
	
	IExecutionContext* context = (IExecutionContext *)(handle[1]);
	const ICudaEngine& engine = context->getEngine();
	// input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
	// of these, but in this case we know that there is exactly one input and one output.
	//assert(engine.getNbBindings() == 3);
	void* buffers[3];

	// In order to bind the buffers, we need to know the names of the input and output tensors.
	// note that indices are guaranteed to be less than IEngine::getNbBindings()
	int inputIndex = engine.getBindingIndex("data"), 
		outputMaskIndex = engine.getBindingIndex("coverage"),
		outputBoxIndex = engine.getBindingIndex("bboxes");
		
	int inputSize  = 1 * 4 * im_h * im_w * sizeof(float);
	int outMaskSize = 1 * mask_h * mask_w * sizeof(float);
	int outBoxSize = 1 * 4 * mask_h * mask_w * sizeof(float);

	// create GPU buffers and a stream
	CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
	CHECK(cudaMalloc(&buffers[outputMaskIndex], outMaskSize));
	CHECK(cudaMalloc(&buffers[outputBoxIndex], outBoxSize));
	
	float *mInputCPU = THFloatTensor_data(input);
	float *mOutMaskCPU = THFloatTensor_data(output_mask);
	float *mOutBoxCPU = THFloatTensor_data(output_box);

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
	CHECK(cudaMemcpyAsync(buffers[inputIndex], mInputCPU, inputSize, cudaMemcpyHostToDevice, stream));
	context->enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(mOutMaskCPU, buffers[outputMaskIndex], outMaskSize, cudaMemcpyDeviceToHost, stream));
	CHECK(cudaMemcpyAsync(mOutBoxCPU, buffers[outputBoxIndex], outBoxSize, cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	// release the stream and the buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputMaskIndex]));
	CHECK(cudaFree(buffers[outputBoxIndex]));
}

