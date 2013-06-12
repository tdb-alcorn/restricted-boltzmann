# a Restricted Boltzmann machine implemented as a weight matrix with state vector, activation energy vector, probability vector and draw from prob vector. State updating has fwd and bwd directions which update the hidden and input nodes respectively.

from mRBMhelper import *

# define the numerical value of on/off states. Setting off to 0 or -1 produces different results which should be interpreted differently. With off=-1, there is a penalty for anti-correlated nodes and a gain for correlated nodes (either on/on or off/off), whereas with off=0 there is only positive reinforcement for the on/on configuration and nothing for the other configurations.
on = 1
off = 0

# make the RBM baby
class RBM:
    "The Restricted Boltzmann Machine"
    def __init__(self,numInputNodes,numHiddenNodes, learningSpeed):
        "defines stuff that you will need, and some helpful variables"
        self.tuning = learningSpeed
        self.nH = numHiddenNodes
        self.Hin = range(1,self.nH+1) # indices corresponding to the hidden nodes in the state vector self.State
        self.nI = numInputNodes
        self.Iin = range(self.nH+1,self.nI+self.nH+1) # indices corresponding to the visible nodes in the state vector self.State
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
        self.connections = symmetrize(self.connections) # now an adjacency matrix C for the RBM. C_{i,j} = 1 if node i is connected to node j, and C_{i,j} = 0 otherwise. I.e. a 
    
    def fwdOutput(self, inputStates):
        "inputState should be a list of lists of inputs for each input node e.g. if there are 3 input nodes and two desired inputs then write [[1,0,0],[1,0,1]]"
        State = np.matrix(inputStates)
        State = State.T
        # insert dummy "off" states for hidden nodes        
        for i in self.Hin:
            State = np.insert(State,1,off,axis=0)
        # activation energies, W*S = A
        A = -1*self.weights*State
        # probability vector, P = logistic(A)
        P = self.getP(self.Hin, A) #only update the hidden nodes
        # draw new states for the hidden nodes
        State = self.draw(self.Hin, State, P)
        return State, A, P

    def bwdOutput(self, hiddenStates):
        "hiddenState should be a list of lists of inputs for each hidden node e.g. if there are 2 hidden nodes and two desired inputs then write [[1,0],[1,1]]"
        State = np.matrix(hiddenStates)
        State = State.T
        # insert "on" state for bias node and dummy "off" states for input nodes
        State = np.append(State,off*np.ones([self.nI,State.shape[1]]),axis=0)
        # activation energies, W*S = A
        A = -1*self.weights*State
        # probability vector, P = logistic(A)
        P = self.getP(self.Iin, A) #only update the input nodes
        # draw new states for the input nodes
        State = self.draw(self.Iin, State, P)
        return State, A, P

    def fwdUpdate(self, inputStates):
        "Forward update method. Takes some visible node states and propagates them forward to the hidden nodes. Also saves self.positive, which is later used in the weight update step"
        self.State, self.A, self.P = self.fwdOutput(inputStates)
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
        "Backward update method. Takes some hidden node states and propagates them back to the visible nodes"
        self.State, self.A, self.P = self.bwdOutput(hiddenStates)
        # turn the hidden states into a nested list for input into bwdUpdate
        nIS = self.State[[0]+self.Iin,:]
        newInputStates = [[0 for i in range(nIS.shape[0])] for j in range(nIS.shape[1])]
        for i in range(nIS.shape[0]):
            for j in range(nIS.shape[1]):
                newInputStates[j][i] = nIS[i,j]
        return newInputStates

    def Train(self, inputStates, numEpochs):
        "This method trains the RBM using training data (inputStates). Training is repeated numEpochs times."
        #readjust tuning to normalize for number of inputstates
        tuning = self.tuning/len(inputStates)
        # insert 1's at start of each input for bias node
        inputStates = [[1]+state for state in inputStates]
        for i in range(numEpochs):
            #raw_input()
            # run fwdUpdate
            hiddenStates = self.fwdUpdate(inputStates)
            # run bwdUpdate
            newInputStates = self.bwdUpdate(hiddenStates)
            #run fwdOutput and save self.negative
            self.fwdOutput(newInputStates)
            # backwards correlation between nodes
            self.negative = self.State*self.State.T
            # the update calculation dW = L*(pos-neg)
            weightUpdates = tuning*(self.positive-self.negative)
            # triangularize the relevant matrices to speed up the multiplication
            self.weights = triangularize(self.weights)
            self.connections = triangularize(self.connections)
            # kill out weightUpdates that correspond to non-existent edges in the graph
            weightUpdates = np.multiply(weightUpdates,self.connections)
            # update the weights
            self.weights += weightUpdates
            # symmetrize the matrices again
            self.weights = symmetrize(self.weights)
            self.connections = symmetrize(self.connections)

    def getP(self, indices, A):
        "Calculates probabilities from activation energies using the logistic function. self.A is an m-by-1 vector"
        P = 0*A
        for i in [0]+indices:
            for j in range(A.shape[1]):
                P[i,j] = 1/(1+2.71818**A[i,j])
        return P

    def draw(self, indices, State, P):
        "Randomly turns nodes on or off using probability vector. self.P is an m-by-1 vector"
        for i in [0]+indices:
            for j in range(P.shape[1]):
                draw = random()
                if draw > P[i,j]:
                    State[i,j] = off
                else:
                    State[i,j] = on
        return State

    def theWeights(self):
        "prints out the weights prettily"
        namesRow = ["Bias"]+["I#"+str(i-self.Iin[0]+1) for i in self.Iin]
        namesCol = ["Bias"]+["H#"+str(i) for i in self.Hin]
        return DataFrame(matRound(self.weights,3)[[0]+self.Iin,:][:,[0]+self.Hin],namesRow,namesCol)

    def FreeEnergy(self, inputStates):
        #From Hinton (2010): The free energy of a data vector can be computed in a time that is linear in the number of hidden units (see section 16.1). If the model is not overfitting at all, the average free energy should be about the same on training and validation data. As the model starts to overfit the average free energy of the validation data will rise relative to the average free energy of the training data and this gap represents the amount of overfitting.
        "Method to quickly compute free energy of a data-set (from Hinton): $F(v) = - \sum_i(v_i a_i) - \sum_j(\log(1+e^{x_j}))$ where $a_i$ are the connections between the visible nodes and the bias node, $x_j$ are the activation energies of the hidden nodes given the state $v$"
        # insert an "off" state for the bias node
        inputStates = [[0]+state for state in inputStates]
        State, A, P = self.fwdOutput(inputStates) # get A
        vDota = A[0,:]
        x = A[self.Hin,:]
        x2 = -1*np.sum(np.log(np.add([1],np.power([2.71818],x))),axis=0)
        F = -1*vDota+x2
        return F


