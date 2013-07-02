
import numpy as np
from scipy.io import loadmat


def sigmoid(x):
    """
    Sigmoid thresholding nonlinear function. Parallelizes over whole array arguments.
    """
    return 1./(1 + np.exp(-x))


on, off = 1., 0.



class RBM(object):
    def __init__(self, nv, nh, weights = None, biases = None):
        self.nv = nv
        self.nh = nh
 
        if biases is not None:
            self.biases = biases
        else:
            self.biases = np.zeros(nv + nh)

        # only have symmetric coupling between visible and hidden
        # layers
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.zeros((nv, nh))
        

 
    @property
    def biases_v(self):
        return self.biases[:self.nv]

    @property
    def biases_h(self):
        return self.biases[self.nv:]

 
    def energy(self, state_v, state_h):
        """
        Compute the current energy function. If the input state is a whole
        batch of states, this returns a vector where each elements
        corresponds to the free energy of the state in the minibatch.
        """
        return -(self.biases_v.dot(state_v) + self.biases_h.dot(state_h) + (state_v * self.weights.dot(state_h)).sum(axis=0))


    def prob_h_on(self, state_v):
        if state_v.ndim == 1:
            return sigmoid(self.biases_h + self.weights.T.dot(state_v))
        else:
            return sigmoid(self.biases_h.reshape((self.nh, 1)) + self.weights.T.dot(state_v))
    

    def prob_v_on(self, state_h):
        if state_h.ndim == 1:
            return sigmoid(self.biases_v + self.weights.dot(state_h))
        else:
            return sigmoid(self.biases_v.reshape((self.nv, 1)) + self.weights.dot(state_h))
    
    
    def update_weights(self, training_batch, epsilon=.05, n_gibbs=1, propagate_probs_v=True, propagate_probs_h=False):
        "This basically implements Hinton's Contrastive Divergence algorithm for training RBMs."
        
        # make training batch 2 dimensional vector
        if training_batch.ndim == 1:
            training_batch.reshape((-1, 1))
            n_batch = 1
        else:
            n_batch = training_batch.shape[1]
        
            
        assert training_batch.shape[0] == self.nv
        state_v = np.array(training_batch)
        

        prob_h_on = self.prob_h_on(state_v)
        
        if propagate_probs_h:
            state_h = prob_h_on
        else:
            state_h = (np.random.rand(self.nh, n_batch) > prob_h_on) * on
        
        dw_data = np.average(state_v.reshape((self.nv, 1, n_batch)) * state_h.reshape((1, self.nh, n_batch)), axis=2)
        db_data = np.hstack([np.average(state_v, axis=1), np.average(state_h, axis=1)])
        
        for k in range(n_gibbs):
            prob_v_on = self.prob_v_on(state_h)
            if propagate_probs_v:
                state_v = prob_v_on
            else:
                state_v = (np.random.rand(self.nv, n_batch) > prob_v_on) * on
            
            prob_h_on = self.prob_h_on(state_v)
            if propagate_probs_h:
                state_h = prob_h_on
            else:
                state_h = (np.random.rand(self.nh, n_batch) > prob_h_on) * on
            
        dw_recon = np.average(state_v.reshape((self.nv, 1, n_batch)) * state_h.reshape((1, self.nh, n_batch)), axis=2)
        db_recon = np.hstack([np.average(state_v, axis=1), np.average(state_h, axis=1)])
        
        self.weights += epsilon * (dw_data - dw_recon)
        self.biases += epsilon * (db_data - db_recon)
        
        return state_v, state_h
        


def main():
    N_train = 2000
    N_test = 300
    
    image_dim = 512
    patch_dim = 16
    num_imgs = 10
    patch_size = patch_dim ** 2

    images = loadmat('IMAGES.mat')['IMAGES']
    
    train_data = np.zeros((N_train, patch_size))
    test_data = np.zeros((N_test, patch_size))

    for kk in range(N_train):
        kr = np.random.randint(0, num_imgs)
        xr, yr =  np.random.randint(0, image_dim - patch_dim, size=2)
        train_data[kk] = images[xr:xr + patch_dim, yr: yr + patch_dim, kr].ravel()
        
    for kk in range(N_test):
        kr = np.random.randint(0, num_imgs)
        xr, yr =  np.random.randint(0, image_dim - patch_dim, size=2)
        test_data[kk] = images[xr:xr + patch_dim, yr: yr + patch_dim, kr].ravel()
        
    # normalize data and binarize
    train_data -= np.mean(train_data, axis=1).reshape((N_train, 1))
    train_data /= np.var(train_data, axis=1).reshape((N_train, 1))
    train_data = 1. * (train_data >= 0)

    nv = patch_size
    nh = 20


    
    test_data -= np.mean(test_data, axis=1).reshape((N_test, 1))
    test_data /= np.var(test_data, axis=1).reshape((N_test, 1))
    test_data = 1. * (test_data >= 0)
    
    p_on = np.average(train_data, axis=0)
    bias_v0 = np.log(p_on/(1-p_on))
    bias_h0 = np.zeros(nh)
    biases = np.hstack((bias_v0, bias_h0))
    weights = np.random.randn(nv, nh) * .01

    rbm = RBM(nv, nh, weights=weights, biases=biases)
    
    batch_size = 10
    epochs = 20000
    
    traj = (np.random.permutation(epochs * batch_size) % N_train).reshape((epochs, batch_size))
    for idx in traj:
        training_batch = train_data[idx]
        rbm.update_weights(training_batch.T)
    
    np.savez('rbm.npz', weights=rbm.weights, biases=rbm.biases)
    return rbm

if __name__ == '__main__':
    rbm = main()

