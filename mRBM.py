# a Restricted Boltzmann machine implemented as a weight matrix with state vector, activation energy vector, probability vector and draw from prob vector. State updating has fwd and bwd directions which update the hidden and input nodes respectively.

import numpy as np
from pandas import DataFrame
from random import random, shuffle

on = 1
off = 0

class RBM:
    "The Restricted Boltzmann Machine"
    def __init__(self,numInputNodes,numHiddenNodes, learningSpeed):
        self.tuning = learningSpeed
        self.nH = numHiddenNodes
        self.Hin = range(1,self.nH+1)
        self.nI = numInputNodes
        self.Iin = range(self.nH+1,self.nI+self.nH+1)
        self.connections = np.ones([self.nH,self.nI])
        self.weights = np.matrix(np.random.randn(self.nH,self.nI)*0.1)
        for i in range(self.nH+1):
            # add 0 columns for hidden nodes and bias node
            self.weights = np.insert(self.weights,0,0,axis=1)
            self.connections = self.connections.T
            self.connections = np.insert(self.connections,0,[0 for i in self.Hin],axis=0)
            self.connections = self.connections.T
        # add 0 row for bias node
        self.weights = np.insert(self.weights,0,self.tuning,axis=0)
        self.connections = np.insert(self.connections,0,1,axis=0)
        # add 0 rows for input nodes
        self.weights = np.append(self.weights,np.zeros([self.nI,self.nH+self.nI+1]),axis=0)
        self.connections = np.append(self.connections,np.zeros([self.nI,self.nH+self.nI+1]),axis=0)
        # symmetrize the bastard
        self.weights = symmetrize(self.weights)
        self.connections = symmetrize(self.connections)
    def fwdOutput(self, inputStates):
        "inputState should be a list of lists of inputs for each input node e.g. if there are 3 input nodes and two desired inputs then write [[1,0,0],[1,0,1]]"
        self.State = np.matrix(inputStates)
        self.State = self.State.T
        # insert "on" state for bias node and dummy "off" states for hidden nodes        
        for i in self.Hin:
            self.State = np.insert(self.State,1,off,axis=0)
        #self.State = np.insert(self.State,0,1,axis=0)
        # activation energies, W*S = A
        self.A = -1*self.weights*self.State
        # probability vector, P = logistic(A)
        self.getP(self.Hin) #only update the hidden nodes
        # draw new states for the hidden nodes
        self.draw(self.Hin)
        return self.State
    def bwdOutput(self, hiddenStates):
        "hiddenState should be a list of lists of inputs for each hidden node e.g. if there are 2 hidden nodes and two desired inputs then write [[1,0],[1,1]]"
        self.State = np.matrix(hiddenStates)
        self.State = self.State.T
        # insert "on" state for bias node and dummy "off" states for input nodes
        self.State = np.append(self.State,off*np.ones([self.nI,self.State.shape[1]]),axis=0)
        #self.State = np.insert(self.State,0,1,axis=0)
        # activation energies, W*S = A
        self.A = -1*self.weights*self.State
        # probability vector, P = logistic(A)
        self.getP(self.Iin) #only update the input nodes
        # draw new states for the input nodes
        self.draw(self.Iin)
        return self.State
    def fwdUpdate(self, inputStates):
        self.fwdOutput(inputStates)
        # forwards correlation between nodes
        self.positive = self.State*self.State.T
        # turn the hidden states into a nested list for input into bwdUpdate
        hS = self.State[[0]+self.Hin,:]
        hiddenStates = [[0 for i in range(hS.shape[0])] for j in range(hS.shape[1])]
        for i in range(hS.shape[0]):
            for j in range(hS.shape[1]):
                hiddenStates[j][i] = hS[i,j]
        return hiddenStates
    def bwdUpdate(self, hiddenStates):
        self.bwdOutput(hiddenStates)
        # turn the hidden states into a nested list for input into bwdUpdate
        nIS = self.State[[0]+self.Iin,:]
        newInputStates = [[0 for i in range(nIS.shape[0])] for j in range(nIS.shape[1])]
        for i in range(nIS.shape[0]):
            for j in range(nIS.shape[1]):
                newInputStates[j][i] = nIS[i,j]
        return newInputStates
    def Train(self, inputStates, numEpochs):
        #readjust tuning to normalize for number of inputstates
        tuning = self.tuning/len(inputStates)
        # insert 1's at start of each input for bias node
        inputStates = [[1]+lesson for lesson in inputStates]
        for i in range(numEpochs):
            #raw_input()
            # run fwdUpdate
            hiddenStates = self.fwdUpdate(inputStates)
            #print "FWD1", self.State, "+", self.positive
            #print "FWD1", self.P
            # run bwdUpdate
            newInputStates = self.bwdUpdate(hiddenStates)
            #print "BWD", self.State
            #print "BWD", self.P
            #run fwdOutput and save self.negative
            self.fwdOutput(newInputStates)
            # backwards correlation between nodes
            self.negative = self.State*self.State.T
            #print "FWD2", self.State, "-", self.negative
            #print "FWD2", self.P
            # the update calculation dW = L*(pos-neg)
            weightUpdates = tuning*(self.positive-self.negative)
            # triangularize the relevant matrices to speed up the multiplication
            self.weights = triangularize(self.weights)
            self.connections = triangularize(self.connections)
            # kill out weightUpdates that correspond to non-existent edges in the graph
            weightUpdates = np.multiply(weightUpdates,self.connections)
            #print "updates", weightUpdates
            # update the weights
            self.weights += weightUpdates
            # symmetrize the matrices again
            self.weights = symmetrize(self.weights)
            self.connections = symmetrize(self.connections)
            #print "weights", self.theWeights()
    def getP(self,indices):
        "Calculates probabilities from activation energies using the logistic function. self.A is an m-by-1 vector"
        self.P = 0*self.A
        for i in [0]+indices:
            for j in range(self.A.shape[1]):
                self.P[i,j] = 1/(1+2.71818**self.A[i,j])
    def draw(self,indices):
        "Randomly turns nodes on or off using probability vector. self.P is an m-by-1 vector"
        for i in [0]+indices:
            for j in range(self.P.shape[1]):
                draw = random()
                if draw > self.P[i,j]:
                    self.State[i,j] = off
                else:
                    self.State[i,j] = on
    def theWeights(self):
        namesRow = ["Bias"]+["I#"+str(i-self.Iin[0]+1) for i in self.Iin]
        namesCol = ["Bias"]+["H#"+str(i) for i in self.Hin]
        return DataFrame(matRound(self.weights,3)[[0]+self.Iin,:][:,[0]+self.Hin],namesRow,namesCol)
        
def matRound(mat,nDigits):
    newMat = np.zeros(mat.shape)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            newMat[i,j] = round(mat[i,j],nDigits)
    return newMat

def symmetrize(a):
    "symmetrizes a U or L triangular square matrix. i.e. either the upper or lower triangle of the square matrix is all zeros, and the other triangle contains the data"
    return a + a.T - np.diag([a[i,i] for i in range(len(a))])

def triangularize(a):
    "turns a square symmetric matrix into an upper triangular matrix"
    for i in range(len(a)):
        for j in range(i):
            a[i,j] = 0
    return a
    
