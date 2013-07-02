# This script tests how varying the number of hidden nodes affects the performance of the network. After messing with these tests a little, I intend to write a new set of methods that make analysing RBM performance easier and more automated. 


from mRBM import *


#
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullLocator
#

# create a 10 by 4 RBM

numInputNodes = 16
numHiddenNodes = 1

# Experiment 1: Examine the performance of the RBM as training continues. Find a way to decide when it has been trained optimally (i.e. not over-trained and not under-trained -> goldilocks zone).

learningRate = 0.05
numEpochs = 100000
trainingData = [[getrandbits(1) for i in range(8)]+[0 for i in range(8)] for i in range(10)]+[[0 for i in range(8)]+[getrandbits(1) for i in range(8)] for i in range(10)]
validationData = [[1 for i in range(8)]+[0 for i in range(8)],[0 for i in range(8)]+[1 for i in range(8)]]

#
UpdateTime = 100
myNet = RBM(numInputNodes,numHiddenNodes,learningRate)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.title('RBM Weight Matrix')
# Reverse the yaxis limits
ax.set_ylim(*ax.get_ylim()[::-1])
enAx = fig.add_subplot(1,2,2)
timeSeries = []
energyDiff = []


for i in range(numEpochs/UpdateTime):
    #setup the weight matrix plot
    ax.clear()
    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())
    # setup the energy time series plot
    # do the calcs
    print myNet.theWeights()
    W = matRound(myNet.weights,3)[[0]+myNet.Iin,:][:,[0]+myNet.Hin].T
    maxWeight = 2**np.ceil(np.log(np.abs(W).max())/np.log(2))
    maxWeight = np.abs(W).max()+1
    for (x,y),w in np.ndenumerate(W):
        if w > 0: color = 'white'
        else:     color = 'black'
        size = np.log10(np.abs(w)+1)/np.log10(maxWeight)+0.05
        rect = Rectangle([x - size / 2, y - size / 2], size, size,
            facecolor=color, edgecolor=color)
        ax.add_patch(rect)
    ax.autoscale_view()
    #
    plt.draw()
    avgFreeTrain = matRound(np.average(np.reshape(myNet.FreeEnergy(trainingData),(-1,2)), axis=0),1)
    freeValid = matRound(myNet.FreeEnergy(validationData),1)
    diffs = freeValid-avgFreeTrain
    print "Avg Free Energies (training data): ", avgFreeTrain
    print "Free Energies (validation data): ", freeValid
    print "Free Energy diff (valid-train): ", diffs
    # train the Net
    myNet.Train(trainingData, UpdateTime,n=i)
    timeSeries += [i*UpdateTime]
    energyDiff += [diffs[0,0]]+[diffs[0,1]]
    eDa = [energyDiff[2*i] for i in range(int(len(energyDiff)/2))]
    eDb = [energyDiff[2*i-1] for i in range(int(len(energyDiff)/2))]
    enAx.clear()
    enAx.plot(timeSeries, eDa, 'r-', label='Free Energy Diff (V-T), Type 1')
    enAx.plot(timeSeries, eDb, 'b-', label='Free Energy Diff (V-T), Type 2')
    enAx.legend()
print myNet.theWeights()

#From Hinton (2010): The free energy of a data vector can be computed in a time that is linear in the number of hidden units (see section 16.1). If the model is not overfitting at all, the average free energy should be about the same on training and validation data. As the model starts to overfit the average free energy of the validation data will rise relative to the average free energy of the training data and this gap represents the amount of overfitting.


# Method to quickly compute free energy of a data-set (from Hinton). --> RBM.FreeEnergy(inputStates)
# It computes F(v) = - \sum_i(v_i a_i) - \sum_j(\log(1+e^{x_j}))


