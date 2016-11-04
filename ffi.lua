caffe = {}
local ffi = require 'ffi'

ffi.cdef[[
void init(void* handle[1],
		  const char* deployFile,
		  const char* modelFile,				 
		  unsigned int maxBatchSize);
void doInference(void* handle[1], THFloatTensor* input, THFloatTensor* output_mask, THFloatTensor* output_box, int batchSize);
]]

gie.C = ffi.load(package.searchpath('libtgie', package.cpath))
