import numpy as np

from layers import *

import pickle
from solver import Solver

class LogisticClassifier(object):
  """
  A logistic regression model with optional hidden layers.
  
  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=100, hidden_dim=None, weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the model. Weights            #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases (if any) using the keys 'W2' and 'b2'.                        #
    ############################################################################
    if hidden_dim is not None:
      W1 = np.random.normal( 0.0, weight_scale, (input_dim,hidden_dim) )
      b1 = np.zeros( (hidden_dim, ) )
      W2 = np.random.normal(0.0, weight_scale, (hidden_dim, 1))
      b2 = np.zeros((1,))

      self.params['W1'] = W1
      self.params['b1'] = b1
      self.params['W2'] = W2
      self.params['b2'] = b2



    else:
      W1 = np.random.normal( 0.0, weight_scale, (input_dim,1) )
      b1 = np.zeros( (1, ) )
      self.params['W1'] = W1
      self.params['b1'] = b1

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

    Inputs:
    - X: Array of input data of shape (N, D)
    - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

    Returns:
    If y is None, then run a test-time forward pass of the model and return:
    - scores: Array of shape (N,) where scores[i] represents the logit for X[i]
    
    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar value giving the loss
    - grads: Dictionary with the same keys as self.params, mapping parameter
      names to gradients of the loss with respect to those parameters.
    """  
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the model, computing the            #
    # scores for X and storing them in the scores variable.                    #
    ############################################################################
    cache = {}
    out, cache['1'] = fc_forward(X, self.params['W1'], self.params['b1'])
    if 'W2' in self.params:
      out, cache['R'] = relu_forward(out)
      out, cache['2'] = fc_forward(out, self.params['W2'], self.params['b2'])
    scores = np.matrix.flatten(out)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the model. Store the loss          #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss and make sure that grads[k] holds the gradients for self.params[k]. #
    # Don't forget to add L2 regularization.                                   #
    #                                                                          #
    ############################################################################
    loss, dout = logistic_loss(scores, y)
    dout = dout.reshape(dout.shape[0],1)
    if 'W2' in self.params:
      loss = loss + 0.5*self.reg * np.linalg.norm(self.params['W2']) ** 2
      dout, grads['W2'], grads['b2'] = fc_backward(dout, cache['2'])
      dout = relu_backward(dout,cache['R'])
      grads['W2'] += self.reg*self.params['W2']

    loss = loss + 0.5*self.reg * np.linalg.norm(self.params['W1']) ** 2
    dout, grads['W1'], grads['b1'] = fc_backward(dout, cache['1'])
    grads['W1'] += self.reg * self.params['W1']
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

input,labels = None,None
with open('data.pkl', 'rb') as f:
    input, labels = pickle.load(f, encoding='latin1')


data = {}
data['X_train'] = input[:500][:]
data['y_train'] = labels[ :500 ]
data['X_val'] = input[500:750][:]
data['y_val'] = labels[ 500:750 ]
data['X_test'] = input[750:][:]
data['y_test'] = labels[ 750: ]
logistic_model = LogisticClassifier(input_dim=data['X_train'].shape[1], hidden_dim=None, weight_scale=1e-3, reg=0.05)
s = Solver(logistic_model, data,
           update_rule='sgd',
           optim_config={
             'learning_rate': 1000e-3,
           },
           lr_decay=0.98,
           num_epochs=50, batch_size=20,
           print_every=100)

s.train()

acc = s.check_accuracy(data['X_test'], data['y_test'], batch_size=20)
print(acc)