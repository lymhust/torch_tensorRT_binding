require 'gie'
require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

prototxt_name = './mnist.prototxt'
binary_name = './mnist.caffemodel'
data_file = '/home/autopilot/gie_samples/samples/data/samples/mnist/'
im_name = '7.pgm'

net = gie.Net(prototxt_name, binary_name)
print('Init OK')

input = image.load(data_file..im_name)*255
img_mean = image.load(data_file..'mean.jpg')*255
input = input - img_mean
output = net:inference(input)
print(output)
print('Inference OK')

_, ind = output:max(1)
print('Image name: '..im_name)
print('Prediction: '..tostring(ind[1]-1))
