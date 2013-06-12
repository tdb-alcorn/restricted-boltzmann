# implements a Restricted Boltzmann Machine.
# "Boltzmann": 2 layers - input layer and hidden layer (no output layer)
# "Restricted": connections only allowed between layers, not within layers

from restrictedBoltzmannMachine import *
from random import random, shuffle

# set squashingFunction, learning speed and number of nodes in each layer
numInputNodes = 6
names = ["Lion King","Mulan","Pocahontas","Les Miserables","Chicago","The Producers"]
numHiddenNodes = 2
learningSpeed = 0.5 # 1 is fastest, 0 is no learning
squashFun = logistic

# setup the nodes. Always put the Bias node first (it matters)
nodes = [BiasStochNode(squashFun,"Bias")]+[InputStochNode(squashFun,names[i]) for i in range(numInputNodes)]+[StochNode(1, squashFun) for i in range(numHiddenNodes)]

#generate the edges
edges = [Edge(startNode,endNode,random()*0.2-0.1) for startNode in nodes for endNode in nodes if shouldIConnect(startNode,endNode,1)]+[Edge(startNode,nodes[0],random()*0.2-0.1) for startNode in nodes[0:numInputNodes+1]]

# make the net
myNet = RBM(nodes,edges,learningSpeed)

#genInput([1, 1, 1, 0.1, 0.1, 0.1],[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

#inputVals is a list of tuples specifying a list of movie preferences
numTrials = 10
DisneyLover = [[1]+[1,1,1,0,0,0] for i in range(numTrials)]
MusicalDude = [[1]+[0,0,0,1,1,1] for i in range(numTrials)]
inputVals = DisneyLover+MusicalDude

# test with 1 input and expval
#inputVals = [(1,1,1.2)]
#expVals = [(0.9,0.1,0.1)]

#scramble the order of the test cases (randomising helps the learning process)
testOrder = range(len(inputVals))
#shuffle(testOrder)
#shuffle(testOrder)

# set the number of epochs to train over
numEpochs = 1000

# run the training
for i in range(numEpochs):
    for j in testOrder:
        myNet.output(inputVals[j])
        myNet.reconstructVisible()
        myNet.adjustWeights()
        #print myNet.theWeights()

# validation tests
print myNet.theWeights()
