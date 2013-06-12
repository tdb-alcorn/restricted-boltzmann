# This code implements a feed-forward neural net designed to classify inputs into three categories. Each input is a tuple of two numbers which represent the value of certain traits (e.g. length and weight of an animal). The neural net consists of an input layer (2 nodes, one for each number in the input), a hidden layer (4 nodes which sum the inputs according to weights *no of nodes chosen arbitrarily) & an output layer (3 nodes, one for each category). The input layer nodes are connected by edges to the hidden layer, with each edge assigned a certain weight. The hidden layer nodes are similarly connected to the output nodes. An input is categorised by whichever output node has the highest value after the input has been completely processed by the neural net. 

from random import random, shuffle

class Net:
    "The neural net, blackboxed."
    def __init__(self, nodes, edges, tuning):
        self.tuning = tuning
        self.edges = edges
        self.nodes = nodes
        self.inputNodes = [node for node in nodes if node.layer==0]
        self.outputNodes = [node for node in nodes if node.layer==-1]
        self.assignEdges()
    def assignEdges(self):
        for node in self.nodes:        
            node.backEdges = [edge for edge in self.edges if edge.end == node]
            node.fwdEdges = [edge for edge in self.edges if edge.start == node]
    def output(self, inputVals):
        self.initialise(inputVals)
        return [node.val() for node in self.outputNodes]
    def initialise(self, inputVals):
        if len(inputVals) != len(self.inputNodes):
            print "Net.initialise(): wrong number of input values."
            pass
        for i in range(len(inputVals)):
            self.inputNodes[i].initialise(inputVals[i])
        return
    def adjustWeights(self,expVals):
        "should only be used after output has been run (at least)  once"
        self.initExpVals(expVals)
        for edge in self.edges:
            edge.weight = edge.weight + self.tuning*edge.start.val()*edge.end.error()
        return [edge.weight for edge in self.edges]
    def initExpVals(self, expVals):
        # assign expVals to output nodes        
        if len(expVals) != len(self.outputNodes):
            print "Net.getErrors(): wrong number of expected values."
            pass
        for i in range(len(expVals)):
            self.outputNodes[i].expect(expVals[i])
        return
    def theWeights(self):
        for node in self.inputNodes:
            print node
            for edge in node.fwdEdges:
                print (edge.end,edge.weight)

class Node:
    "A node of the neural net, connected to other nodes. Has a list of edges that end with this node, Node.edges. Has a layer attribute Node.layer, which is a number specifying what layer the node is in (key: 0=input, 1=hidden, -1=output). Has an squashing function g(v) Node.squash. Usually set to the logistic function"
    def __init__(self, layer, squashingFunction):
        self.layer = layer
        self.squash = squashingFunction
    def val(self):
        v = 0
        for edge in self.backEdges:
            v += edge.start.val()*edge.weight
        return self.squash(v)
    def error(self):
        v = self.val()
        e = 0
        for edge in self.fwdEdges:
            e += edge.end.error()*edge.weight
        return v*(1-v)*e


class OutputNode(Node):
    "the output nodes"
    def __init__(self, squashingFunction):
        Node.__init__(self,-1,squashingFunction)
        #self.layer = -1
    def expect(self,expVal):
        self.expVal = expVal
    def error(self):
        y = self.expVal
        v = self.val()
        return v*(1-v)*(y-v)


class InputNode(Node):
    "The input nodes"
    def __init__(self, squashingFunction):
        Node.__init__(self, 0, squashingFunction)        
        #self.layer = 0
    def initialise(self, initVal):
        self.initVal = initVal
    def val(self):
        return self.initVal
    def error(self):
        return 0

class BiasNode(InputNode):
    "the bias node. it's just an input where you should always feed it 1"
    def initialise(self, initVal):
        if initVal!=1: 
            print "BiasNode.initialise: Bias node should have inputVal=1, you knave"
            pass
        else:
            self.initVal = initVal


class Edge:
    "A directed weighted edge of the neural net, connecting two nodes. Edge weight is given by Edge.weight. Has a StartNode attribute, Edge.start, that points to the starting node. Similarly, there is an Edge.end attribute."
    def __init__(self, start, end, weight):
        self.weight = weight
        self.start = start
        self.end = end


def logistic(x):
    e = 2.71818
    return 1/(1+e**(-1*x))


def classify(outputNodes):
    "this function is not fully implemented yet. In particular, node.name isn't a thing right now (6/7/13)"
    ans = ("you fucked up",-10^6)
    for node in outputNodes:
        val = node.val()
        if val > ans[1]:
           ans = (node.name, val)
    return ans


def shouldIConnect(startNode,endNode,numLayers):
    "says whether or not nodes are in adjacent layers and therefore should be connected. numLayers is the number of HIDDEN layers (input and output layers aren't counted)"
    if endNode.layer==-1 and startNode.layer==numLayers:
            return True
    elif endNode.layer-startNode.layer==1:
        return True
    return False

def genInput(center, spread):
    if len(center)!=len(spread):
        print "genInput: center and spread lists are not same length."
        pass
    else:
        return [center[i]+random()*spread[i]*2-spread[i] for i in range(len(center))]
