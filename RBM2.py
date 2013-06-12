from mRBM import *

numInputNodes = 6
numHiddenNodes = 2
learningRate = 0.05

numEpochs = 10000
#training = [[1,1,1,0,0,0],[1,0,1,0,0,0],[1,1,1,0,0,0],[0,0,1,1,0,0],[0,0,1,1,1,0]]
training = [[1,1,1,0,0,0],[0,0,1,1,1,0]]
#training = [[1,1,1,0,0,0]]

myNet = RBM(numInputNodes,numHiddenNodes,learningRate)
myNet.Train(training, numEpochs)
print myNet.theWeights()

