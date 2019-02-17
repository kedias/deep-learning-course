import numpy as np
import pickle
from layers import *
from solver import Solver

class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool - fc - softmax

  You may also consider adding dropout layer or batch normalization layer. 
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(1, 28, 28), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0, pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2},
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    W1 = np.random.normal(0, weight_scale, (num_filters,input_dim[0],filter_size,filter_size))
    b1 = np.zeros((1,num_filters,filter_size,filter_size))
    Hn = int(np.floor(1. + (input_dim[1] - filter_size + 1 - pool_param['pool_height']) / pool_param['stride']))
    Wn = int(np.floor(1. + (input_dim[2] - filter_size + 1 - pool_param['pool_width']) / pool_param['stride']))
    self.params['gamma'] = np.ones( (num_filters*Hn*Wn,) )
    self.params['beta'] = np.zeros( (num_filters*Hn*Wn,) )
    W2 = np.random.normal(0, weight_scale, (num_filters*Hn*Wn, hidden_dim))
    b2 = np.zeros((hidden_dim,))
    W3 = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
    b3 = np.zeros((num_classes,))

    self.params['W1'] = W1
    self.params['W2'] = W2
    self.params['W3'] = W3
    self.params['b1'] = b1
    self.params['b2'] = b2
    self.params['b3'] = b3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    mode = 'train'
    if y is None:
      mode = 'test'

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    cache = {}
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    out, cache['0'] = conv_forward(X, W1)
    out, cache['1'] = relu_forward(out)
    out, cache['2'] = max_pool_forward( out, pool_param )
    N,C,H,W = out.shape
    out = out.reshape(N, C*H*W)
    out, cache['B'] = batchnorm_forward( out,self.params['gamma'],self.params['beta'], {'eps':1e-5, 'momentum':0.9, 'mode': mode} )
    out, cache['D'] = dropout_forward(out, {'p':0.5, 'mode': mode})
    out, cache['3'] = fc_forward( out,W2,b2 )
    out, cache['4'] = relu_forward(out)
    out, cache['5'] = fc_forward(out, W3, b3)

    scores = out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dout = softmax_loss( scores,y )
    dout,grads['W3'],grads['b3'] = fc_backward(dout,cache['5'])
    dout = relu_backward( dout,cache['4'] )
    dout, grads['W2'], grads['b2'] = fc_backward(dout, cache['3'])
    dout = dropout_backward( dout,cache['D'] )
    dout, grads['gamma'], grads['beta'] = batchnorm_backward(dout, cache['B'])
    dout = dout.reshape(N,C,H,W)
    dout = max_pool_backward(dout,cache['2'])
    dout = relu_backward(dout,cache['1'])
    dout, grads['W1'] = conv_backward(dout, cache['0'])
    grads['b1'] = 0
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
data['X_train'] = data['X_train'].reshape( 50000,1,28,28 )
data['X_val'] = data['X_val'].reshape( 10000,1,28,28 )
data['X_test'] = data['X_test'].reshape( 10000,1,28,28 )


model=ConvNet(input_dim= (1,28,28), num_filters=16, hidden_dim=100, filter_size=7)
s = Solver(model, data,
                update_rule='sgd',
                optim_config={
                'learning_rate': 200e-3,
                },
                lr_decay=0.95,
                num_epochs=1, batch_size=20,
                print_every=100)
s.train()
acc=s.check_accuracy(data['X_test'], data['y_test'],batch_size=20)
print(acc)
