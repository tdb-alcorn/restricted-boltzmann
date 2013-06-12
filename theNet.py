# This code implements a feed-forward neural net designed to classify inputs into three categories. Each input is a tuple of two numbers which represent the value of certain traits (e.g. length and weight of an animal). The neural net consists of an input layer (2 nodes, one for each number in the input), a hidden layer (4 nodes which sum the inputs according to weights *no of nodes chosen arbitrarily) & an output layer (3 nodes, one for each category). The input layer nodes are connected by edges to the hidden layer, with each edge assigned a certain weight. The hidden layer nodes are similarly connected to the output nodes. An input is categorised by whichever output node has the highest value after the input has been completely processed by the neural net. 

from neuralNet import *
from random import random, shuffle

# set squashingFunction and number of nodes in each layer (1 hidden layer only)
numInputNodes = 2
numOutputNodes = 3
numHiddenNodes = 4
squashFun = logistic

# setup the nodes
nodes = [BiasNode(squashFun)]+[InputNode(squashFun) for i in range(numInputNodes)]+[Node(1, squashFun) for i in range(numHiddenNodes)]+[OutputNode(squashFun) for i in range(numOutputNodes)]

#generate the edges
edges = [Edge(startNode,endNode,random()*0.1-0.05) for startNode in nodes for endNode in nodes if shouldIConnect(startNode,endNode,1)]

# make the net
myNet = Net(nodes,edges,0.5)

#inputVals is a list of tuples specifying (length,weight) of an animal
numTrials = 10
cheetahs = [(1,1+random()*0.2-0.1,1.2+(random()*0.1-0.05)) for i in range(numTrials)]
lions = [(1,1.5+random()*0.2-0.1,0.8+(random()*0.1-0.05)) for i in range(numTrials)]
tigers = [(1,2+random()*0.2-0.1,1.6+(random()*0.1-0.05)) for i in range(numTrials)]
inputVals = cheetahs+lions+tigers

#expVals is list of tuples specifying (cheetah, lion, tiger)
expVals = [(0.9,0.1,0.1) for i in cheetahs]+[(0.1,0.9,0.1) for i in lions]+[(0.1,0.1,0.9) for i in tigers]

# test with 1 input and expval
#inputVals = [(1,1,1.2)]
#expVals = [(0.9,0.1,0.1)]

#scramble the order of the test cases (randomising helps the learning process)
testOrder = range(len(inputVals))
shuffle(testOrder)
shuffle(testOrder)

# set the number of epochs to train over
numEpochs = 1000

# run the training
for i in range(numEpochs):
    for j in testOrder:
        myNet.output(inputVals[j])
        myNet.adjustWeights(expVals[j])

# validation tests
print "should be cheetah"
print myNet.output((1,1,1.2))
print "should be lion"
print myNet.output((1,1.5,0.8))
print "should be tiger"
print myNet.output((1,2,1.6))


