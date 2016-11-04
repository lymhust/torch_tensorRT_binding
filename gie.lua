local Net, parent = torch.class('gie.Net', 'nn.Module')
local ffi = require 'ffi'
local C = gie.C

function Net:__init(deployFile, modelFile, im_h, im_w)

	assert(type(deployFile) == 'string')
	assert(type(modelFile) == 'string')
	assert(type(im_h) == 'number')
	assert(type(im_w) == 'number')

	parent.__init(self)
	self.handle = ffi.new'void*[1]'
	local old_handle = self.handle[1]
    C.init(self.handle, deployFile, modelFile, 1)	
	if(self.handle[1] == old_handle) then
		print 'Unsuccessful init'
	end
	
	local im_h_s = torch.round(im_h/16)
	local im_w_s = torch.round(im_w/16)
	self.output_mask = torch.Tensor(1,im_h_s,im_w_s):zero()
	self.output_box = torch.Tensor(1,4,im_h_s,im_w_s):zero()
	self:float()
end

function Net:inference(input)
	assert(input:type() == 'torch.FloatTensor')
	C.doInference(self.handle, input:cdata(), self.output_mask:cdata(), self.output_box:cdata(), 1);
	return self.output_mask, self.output_box
end
