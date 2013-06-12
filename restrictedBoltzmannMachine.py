# implements a Restricted Boltzmann Machine.
# see neuralNet.py for the origin of these classes

from neuralNet import *
from random import random, shuffle
from pandas import DataFrame



class RBM(Net):
    "the RBM, blackboxed"
    def __init__(self, nodes, edges, tuning):
        Net.__init__(self,nodes,edges,tuning)
        self.outputNodes = [node for node in nodes if node.layer==1]+[nodes[0]] # because there is no output layer, only 1 hidden layer which effectively constitutes the output layer
    def output(self, inputVals):
        outputs = Net.output(self, inputVals)
        for edge in self.edges:
            edge.positive = edge.start.val()[0]*edge.end.val()[0]
        return outputs[0:-1]
    def reconstructVisible(self):
        outputs = [node.backVal() for node in self.inputNodes]
        for edge in self.edges:
            edge.negative = edge.start.val()[0]*edge.end.val()[0]
        return outputs
    def adjustWeights(self):
        "should only be used after output has been run (at least) once"
        for edge in self.edges:
            edge.weight = edge.weight + self.tuning*(edge.positive - edge.negative)
        return [edge.weight for edge in self.edges]
    def theWeights(self):
        weightMatrix = []
        names = []
        for node in self.inputNodes:
            names += [node.name]
            weights = []
            for edge in node.fwdEdges:
                weights+=[edge.weight]
            weightMatrix+=[weights]
        #print weightMatrix
        return DataFrame(weightMatrix,names,["Bias"]+[str(i+1) for i in range(len(weightMatrix[0])-1)])


class StochNode(Node):
    def val(self):
        v = 0
        for edge in self.backEdges:
            v += edge.start.val()[1]*edge.weight
        value = self.squash(v)
        draw = random()
        if draw > value:
            self.on = 0
        else:
            self.on = 1
        return (value, self.on)


class InputStochNode(StochNode):
    "The input nodes"
    def __init__(self, squashingFunction,name):
        Node.__init__(self, 0, squashingFunction)        
        #self.layer = 0
        self.name = name
    def initialise(self, initVal):
        self.initVal = initVal
    def val(self):
        return [self.initVal,self.initVal]
    def backVal(self):
        v = 0
        for edge in self.fwdEdges:
            v += edge.end.val()[1]*edge.weight
        val = self.squash(v)
        draw = random()
        if draw > val:
            self.on = 0
        else:
            self.on = 1
        return (val, self.on)

class BiasStochNode(InputStochNode):
    "the bias node. it's just an input where you should always feed it 1"
    def initialise(self, initVal):
        if initVal!=1: 
            print "BiasNode.initialise: Bias node should have inputVal=1, you knave"
            pass
        else:
            self.initVal = initVal
    

