# Forward Pass CNN
# INPUT => CONV => RELU => POOL => CONV => RELU => POOL => FC(Conv) => RELU => FC
import sys
import numpy as np
from PIL import Image
import os
ImgRowSize = 32
ImgColSize = 32
FilterRowSize = 5
FilterColSize = 5
FilterDepth1 = 3
FilterDepth2= 6
FilterDepth3=16
FilterCount1=6
FilterCount2=16
FilterCount3=120
PoolingRowSize = 2
PoolingColSize = 2
# FilterDepthFcc = 120
NNInputSize = 120
NNHiddenUnits = 84
NNOutputSize = 10

#Minimum 1 input check
if len(sys.argv) < 2:
	print("Enter the path of Image in Command Line Arguments!")
	sys.exit(1)

def convolveOperation(data,dataFilter):
	convRes = np.multiply(data,dataFilter)
	return convRes.sum()

def image_show(data):
	img1 = Image.fromarray(data,'RGB')
	img1 = img1.resize((312,312))
	img1.show()

def convolve(data,datafilter):
	dataRow, dataCol, dataChannels = data.shape
	filterRow, filterCol, filterdepth,filtercount = datafilter.shape
	# print(datafilter.shape)
	convResultDim = dataRow - filterRow + 1
	convResult = np.zeros((convResultDim,convResultDim,filtercount))
	for filNum in range(filtercount):
		for x in range(convResultDim):
			for y in range(convResultDim):
				convResult[x][y][filNum] = convolveOperation(data[x:x+filterRow,y:y+filterCol,:],datafilter[:,:,:,filNum])
	return convResult

def createFilter(depth,filtercount):
	filterMatrix = np.random.rand(FilterRowSize,FilterColSize,depth,filtercount)*0.01
	# filterMatrix=np.full((FilterRowSize,FilterColSize,depth,filtercount),1)
	return filterMatrix

def ReLU(x):
	return np.where(x>0,np.where(x<255.0,x,255),0.0)

def forwardPass(i):
	weightInputHidden = np.random.randn(NNInputSize,NNHiddenUnits)
	weightOutputHidden = np.random.randn(NNHiddenUnits,NNOutputSize)
	z = np.dot(i,weightInputHidden)
	act = sigmoid(z)
	z1 = np.dot(act,weightOutputHidden)
	act2 = sigmoid(z1)
	return act2

def maxPooling(data,poolRowSize,poolColSize,stride):
	dataRow, dataCol, resLayer = data.shape
	# print("dataRow ",dataRow)
	# print("poolRowSize ",poolRowSize)
	# print("Inside maxpool ",data.shape)
	# print("poolres row ",int((dataRow-poolRowSize)/stride)+1)
	# print("poolres col ",int((dataCol-poolColSize)/stride)+1)
	# print("Stride ",stride)
	poolResRow=int((dataRow-poolRowSize)/stride)+1
	poolResCol=int((dataCol-poolColSize)/stride)+1
	poolRes = np.zeros((poolResRow,poolResCol,resLayer))
	# print("data ",data)
	for l in range(resLayer):
		i = 0
		while i < dataRow:
			r=stride
			j = 0
			while j < dataCol and i+poolRowSize<dataRow and j+poolColSize<dataCol:
					x=int(i/2)
					y=int(j/2)
					dep=l
					poolRes[x,y,l] = np.max(data[i:i+poolRowSize,j:j+poolColSize,l])
					j =j+r
			i=i+r
	return poolRes

def sigmoid(x):
	return 1 / (1+np.exp(-x))

IMG_PATH = sys.argv[1]
if os.path.isfile(IMG_PATH):
	img = Image.open(IMG_PATH)
	a1 = np.array(img)
	print("Original Dimensions of the Image:",a1.shape)
	img = img.resize((ImgRowSize,ImgColSize),Image.ANTIALIAS) #A high Quality downsampling is done by image.AntiAlias
	imageArray = np.array(img)
	print("Reduced Dimensions of the Image:",imageArray.shape)
else :
	print("Couldn't find the given image. Try again!")
	sys.exit(1)

# ~~~~~~~~~~~~~~~~ 1st BLOCK ~~~~~~~~~~~~~~~(Each block Conv-->Relu--->maxpool)
filterL1 = createFilter(FilterDepth1,FilterCount1)  #5x5x3
convResult = convolve(imageArray,filterL1)
print("Dimensions after 1st Convolution:",convResult.shape)
reLURes = ReLU(convResult)
image_show(reLURes)
poolResult = maxPooling(reLURes,PoolingRowSize,PoolingColSize,2)
image_show(poolResult)
print("Dimensions after 1st Max Pooling:",poolResult.shape)   #14x14x6
# ~~~~~~~~~~~~~~~~~~~ 2nd Block ~~~~~~~~~~~~~~~~~~~~~
filterL2 = createFilter(FilterDepth2,FilterCount2) #5x5x6
convResult2 = convolve(poolResult,filterL2)
print("Dimension after 2nd Convolution:",convResult2.shape)
reLURes2 = ReLU(convResult2)
image_show(reLURes2)
poolResult2 = maxPooling(reLURes2,PoolingRowSize,PoolingColSize,2) #5x5x16
image_show(poolResult2)
print("Dimension after 2nd Max Pooling:",poolResult2.shape)
# ~~~~~~~~~~~~~~~~~ CONVOLUTION AT FIRST FC LAYER ~~~~~~~~~~~~~~~~~(final layer: conv-->Relu-->forwardPass-->softmax)
filterL3 = createFilter(FilterDepth3,FilterCount3) #505016
print("Filter Dimensions for FC Convolution:",filterL3.shape)
convResult3 = convolve(poolResult2,filterL3)
print("Dimensions after FC Convolution:",convResult3.shape)
reLURes3 = ReLU(convResult3)
image_show(reLURes3)
nnOutput = forwardPass(reLURes3)
print("Dimensions of Output:",nnOutput.shape)
# print("nnOutput ",nnOutput)
z=nnOutput[0,0,:]
# print("z ",z)
# print("z1",list(nnOutput))
expScores = np.exp(z)
probs = expScores/np.sum(expScores)
softmaxRes = probs


# print("Possible Number on Image: ")
print(np.argmax(softmaxRes))