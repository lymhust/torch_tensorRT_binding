require 'gie'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

prototxt_name = '/home/autopilot/lymcode/detectnet_deploy/model_car/deploy.prototxt'
binary_name = '/home/autopilot/lymcode/detectnet_deploy/model_car/weights.caffemodel'
data_file = '/home/autopilot/lymcode/detectnet_deploy/2015-03-26-12-51-24_00000.jpg'

local im_w, im_h = 1024, 512
net = gie.Net(prototxt_name, binary_name, im_h, im_w)
print('Init OK')

input = image.scale(image.load(data_file), im_w, im_h):mul(255)
print(#input)

for i = 1, 1 do
	sys.tic()
	mask, box = net:inference(input)
	local tm = sys.toc()
	print('Forward time GIE: '..(tm*1000))
	print('FPS GIE: '..(1/sys.toc()))
end
print(mask)
print(mask:max())
image.save('./mask_result.jpg', mask)
print(#mask)
print(#box)
print('Inference OK')

