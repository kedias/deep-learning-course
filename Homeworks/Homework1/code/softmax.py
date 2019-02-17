import numpy as np
from layers import *
from solver import Solver
import pickle

class SoftmaxClassifier(object):
    """
    A fully-connected neural network with
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecture should be fc - softmax if no hidden layer.
    The architecture should be fc - relu - fc - softmax if one hidden layer

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=28*28, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with fc weights                                  #
        # and biases using the keys 'W' and 'b'                                    #
        ############################################################################
        if hidden_dim is not None:
            self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim) )
            self.params['b1'] = np.zeros( (hidden_dim,) )
            self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
            self.params['b2'] = np.zeros((num_classes,))



        else:
            self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, num_classes))
            self.params['b1'] = np.zeros((num_classes,))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the one-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        cache = {}
        out, cache['0'] = fc_forward( X, self.params['W1'], self.params['b1'] )
        if 'W2' in self.params:
            out, cache['1'] = relu_forward(out)
            out, cache['2'] = fc_forward( out, self.params['W2'], self.params['b2'] )
        scores = out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the one-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss( scores, y )
        if 'W2' in self.params:
            loss = loss + 0.5 * self.reg * np.linalg.norm(self.params['W2']) ** 2
            dout, grads[ 'W2' ], grads['b2'] = fc_backward(dout,cache['2'])
            dout = relu_backward( dout, cache['1'] )
            grads['W2'] += self.reg * self.params['W2']
        loss = loss + 0.5 * self.reg * np.linalg.norm(self.params['W1']) ** 2
        gradx, grads[ 'W1' ],grads['b1'] = fc_backward( dout, cache['0'] )
        grads['W1'] += self.reg * self.params['W1']
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


train, valid, test = None, None,None
with open('mnist.pkl', 'rb') as f:
    train, valid, test = pickle.load(f, encoding='latin1')


data = {}
data['X_train'],data['y_train'] = train
data['X_val'],data['y_val'] = valid
data['X_test'],data['y_test'] = test

model=SoftmaxClassifier(hidden_dim = 100, reg=0.005)
s = Solver(model, data,
                update_rule='sgd',
                optim_config={
                'learning_rate': 100e-3,
                },
                lr_decay=0.95,
                num_epochs=10, batch_size=20,
                print_every=100)
s.train()
acc=s.check_accuracy(data['X_test'],data['y_test'],batch_size=20)
print(acc)
