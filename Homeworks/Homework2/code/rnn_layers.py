import numpy as np

"""
This file defines layer types that are commonly used for recurrent neural
networks.
"""


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """
    Run the forward pass for a single timestep of a vanilla RNN that uses a tanh
    activation function.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data for this timestep, of shape (N, D).
    - prev_h: Hidden state from previous timestep, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - cache: Tuple of values needed for the backward pass.
    """

    ##############################################################################
    # TODO: Implement a single forward step for the vanilla RNN. Store the next  #
    # hidden state and any values you need for the backward pass in the next_h   #
    # and cache variables respectively.                                          #
    ##############################################################################
    next_h = np.tanh( np.matmul(prev_h, Wh) + np.matmul(x, Wx) + b.T)
    cache = next_h, x, prev_h, Wx, Wh

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    """
    Backward pass for a single timestep of a vanilla RNN.
    Inputs:
    - dnext_h: Gradient of loss with respect to next hidden state
    - cache: Cache object from the forward pass
    Returns a tuple of:
    - dx: Gradients of input data, of shape (N, D)
    - dprev_h: Gradients of previous hidden state, of shape (N, H)
    - dWx: Gradients of input-to-hidden weights, of shape (D, H)
    - dWh: Gradients of hidden-to-hidden weights, of shape (H, H)
    - db: Gradients of bias vector, of shape (H,)
    """

    ##############################################################################
    # TODO: Implement the backward pass for a single step of a vanilla RNN.      #
    #                                                                            #
    # HINT: For the tanh function, you can compute the local derivative in terms #
    # of the output value from tanh.                                             #
    ##############################################################################
    next_h, x, prev_h, Wx, Wh = cache
    dtanh = np.ones_like( next_h ) - next_h*next_h
    dnewnexth = dtanh*dnext_h
    dx = np.matmul( dnewnexth,Wx.T )
    dprev_h = np.matmul( dnewnexth,Wh.T )
    db = np.sum( dnewnexth.T,axis=1 )
    dWx = np.matmul( x.T, dnewnexth )
    dWh = np.matmul( prev_h.T, dnewnexth )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.
    Inputs:
    - x: Input data for the entire timeseries, of shape (T, N, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)
    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (T, N, H).
    - cache: Values needed in the backward pass
    """
    h, cache = None, None
    ##############################################################################
    # TODO: Implement forward pass for a vanilla RNN running on a sequence of    #
    # input data. You should use the rnn_step_forward function that you defined  #
    # above. You can use a for loop to help compute the forward pass.            #
    ##############################################################################
    h = np.zeros( (x.shape[0], x.shape[1], h0.shape[1]) )
    cache = [None] * x.shape[0]
    h[0], cache[0] = rnn_step_forward(x[0],h0,Wx,Wh,b)
    for i in range(1,x.shape[0]):
        h[i], cache[i] = rnn_step_forward( x[i],h[i-1],Wx,Wh,b )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.
    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (T, N, H)
    Returns a tuple of:
    - dx: Gradient of inputs, of shape (T, N, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    ##############################################################################
    # TODO: Implement the backward pass for a vanilla RNN running an entire      #
    # sequence of data. You should use the rnn_step_backward function that you   #
    # defined above. You can use a for loop to help compute the backward pass.   #
    ##############################################################################
    next_h, x, prev_h, Wx, Wh = cache[-1]
    dx = np.zeros((dh.shape[0], dh.shape[1], x.shape[1]))
    dnextH = dh[-1]
    dx[-1], dprev_h, dWx, dWh, db = rnn_step_backward(dnextH, cache[-1])

    for i in reversed(range(0, dh.shape[0]-1)):
        dnextH = dh[i] + dprev_h
        dx[i], dprev_h, temp_dWx, temp_dWh, temp_db = rnn_step_backward(dnextH,cache[i])
        dWx += temp_dWx
        dWh += temp_dWh
        db += temp_db

    dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """
    Forward pass for a single timestep of an LSTM.
    The input data has dimension D, the hidden state has dimension H, and we use
    a minibatch size of N.
    Inputs:
    - x: Input data, of shape (N, D)
    - prev_h: Previous hidden state, of shape (N, H)
    - prev_c: previous cell state, of shape (N, H)
    - Wx: Input-to-hidden weights, of shape (D, 4H)
    - Wh: Hidden-to-hidden weights, of shape (H, 4H)
    - b: Biases, of shape (4H,)
    Returns a tuple of:
    - next_h: Next hidden state, of shape (N, H)
    - next_c: Next cell state, of shape (N, H)
    - cache: Tuple of values needed for backward pass.
    """
    cache = None
    #############################################################################
    # TODO: Implement the forward pass for a single timestep of an LSTM.        #
    # You may want to use the numerically stable sigmoid implementation above.  #
    # For Wx of shape D x 4H, you may assume they are the sequence of parameters#
    # for forget gate, input gate, concurrent input, output gate. Wh and b also #
    # follow the same order.                                                    #
    #############################################################################
    H = prev_h.shape[1]
    A = np.matmul(prev_h, Wh) + np.matmul(x, Wx) + b.T
    f = sigmoid(A[:,:H])
    i = sigmoid( A[:,H:2*H] )
    chat = np.tanh(A[:,2*H:3*H])
    o = sigmoid(A[:,3*H:])
    next_c = f * prev_c + i * chat
    next_h = o * np.tanh(next_c)
    cache = prev_h,Wh,x,Wx,b,f,i,chat,o,prev_c,next_h,next_c, H
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return next_h, next_c, cache


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.
    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh, dc, dWx, dWh, db, dprev_h, dprev_c = None, None, None, None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for a single timestep of an LSTM.       #
    #                                                                           #
    # HINT: For sigmoid and tanh you can compute local derivatives in terms of  #
    # the output value from the nonlinearity.                                   #
    #############################################################################
    prev_h,Wh,x,Wx,b,f,i,chat,o,prev_c,next_h,next_c, H = cache

    hc = np.tanh(next_c)
    dhc = o * (np.ones_like(hc) - hc*hc)
    dc = dnext_c + dhc * dnext_h
    dprev_c = f * dc
    do = o * (np.ones_like(o)-o) * hc * dnext_h
    df = f * (np.ones_like(f)-f) * prev_c * dc
    di = i * (np.ones_like(i) - i) * chat * dc
    dchat = (np.ones_like(chat) - chat*chat) * i * dc

    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)

    db[:H] = np.sum( df.T,axis=1 )
    dWx[:,:H] = np.matmul( x.T, df )
    dWh[:,:H] = np.matmul( prev_h.T, df )
    dx = np.matmul( df,Wx[:,:H].T )
    dprev_h = np.matmul( df,Wh[:,:H].T )

    db[H:2*H] = np.sum( di.T,axis=1 )
    dWx[:,H:2*H] = np.matmul( x.T, di)
    dWh[:,H:2*H] = np.matmul( prev_h.T, di)
    dx += np.matmul( di,Wx[:,H:2*H].T )
    dprev_h += np.matmul( di,Wh[:,H:2*H].T )

    db[2*H:3*H] = np.sum( dchat.T,axis=1 )
    dWx[:,2*H:3*H] = np.matmul( x.T, dchat )
    dWh[:,2*H:3*H] = np.matmul( prev_h.T, dchat )
    dx += np.matmul( dchat,Wx[:,2*H:3*H].T )
    dprev_h += np.matmul( dchat,Wh[:,2*H:3*H].T )

    db[3*H:] = np.sum( do.T,axis=1 )
    dWx[:,3*H:] = np.matmul( x.T, do )
    dWh[:,3*H:] = np.matmul( prev_h.T, do )
    dx += np.matmul( do,Wx[:,3*H:].T )
    dprev_h += np.matmul( do,Wh[:,3*H:].T )

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dprev_h, dprev_c, dWx, dWh, db


def lstm_forward(x, h0, Wx, Wh, b):
    """
    Forward pass for an LSTM over an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The LSTM uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the LSTM forward, we return the hidden states for all timesteps.
    Note that the initial cell state is passed as input, but the initial cell
    state is set to zero. Also note that the cell state is not returned; it is
    an internal variable to the LSTM and is not accessed from outside.
    Inputs:
    - x: Input data of shape (T, N, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)
    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (T, N, H)
    - cache: Values needed for the backward pass.
    """
    h, cache = None, None
    #############################################################################
    # TODO: Implement the forward pass for an LSTM over an entire timeseries.   #
    # You should use the lstm_step_forward function that you just defined.      #
    #############################################################################
    h = np.zeros( (x.shape[0], x.shape[1], h0.shape[1]) )
    cache = [None] * x.shape[0]
    h[0], next_c, cache[0] = lstm_step_forward(x[0],h0,np.zeros_like(h0),Wx,Wh,b)
    for i in range(1,x.shape[0]):
        h[i], next_c, cache[i] = lstm_step_forward( x[i],h[i-1],next_c, Wx,Wh,b )
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]
    Inputs:
    - dh: Upstream gradients of hidden states, of shape (T, N, H)
    - cache: Values from the forward pass
    Returns a tuple of:
    - dx: Gradient of input data of shape (T, N, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    #############################################################################
    # TODO: Implement the backward pass for an LSTM over an entire timeseries.  #
    # You should use the lstm_step_backward function that you just defined.     #
    #############################################################################
    _, Wh, x, Wx, b, f, i, chat, o, _, _, _, _ = cache[-1]
    dx = np.zeros((dh.shape[0], dh.shape[1], x.shape[1]))
    dnextH = dh[-1]
    dnextC = np.zeros_like(dnextH)
    dx[-1], dprev_h, dprev_c, dWx, dWh, db = lstm_step_backward(dnextH, dnextC, cache[-1])

    for i in reversed(range(0, dh.shape[0] - 1)):
        dnextH = dh[i] + dprev_h
        dnextC = dprev_c
        dx[i], dprev_h, dprev_c, temp_dWx, temp_dWh, temp_db = lstm_step_backward(dnextH, dnextC, cache[i])
        dWx += temp_dWx
        dWh += temp_dWh
        db += temp_db

    dh0 = dprev_h

    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.
    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x must be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.
    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    out = W[x, :]
    cache = x, W
    return out, cache

def word_embedding_backward(dout, cache):
    """
    Backward pass for word embeddings. We cannot back-propagate into the words
    since they are integers, so we only return gradient for the word embedding
    matrix.
    HINT: Look up the function np.add.at
    Inputs:
    - dout: Upstream gradients of shape (N, T, D)
    - cache: Values from the forward pass
    Returns:
    - dW: Gradient of word embedding matrix, of shape (V, D).
    """
    dW = None
    x, W = cache
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW



def temporal_fc_forward(x, w, b):
    """
    Forward pass for a temporal fully-connected layer. The input is a set of D-dimensional
    vectors arranged into a minibatch of N timeseries, each of length T. We use
    an affine function to transform each of those vectors into a new vector of
    dimension M.
    Inputs:
    - x: Input data of shape (N, T, D)
    - w: Weights of shape (D, M)
    - b: Biases of shape (M,)
    Returns a tuple of:
    - out: Output data of shape (N, T, M)
    - cache: Values needed for the backward pass
    """
    out = np.matmul(x,w) + b.T
    cache = (x, w, b)
    return out,cache


def temporal_fc_backward(dout, cache):
    """
    Backward pass for temporal fully-connected layer.
    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass
    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x,w,b = cache
    dx = np.matmul(dout, w.T)
    dw = np.sum(np.matmul(x.transpose(1,2,0), dout.transpose(1,0,2)), axis = 0)
    db = np.sum(np.sum(dout, axis=0), axis=0).T
    return dx, dw, db


def temporal_softmax_loss(x, y, mask):
    """
    A temporal version of softmax loss for use in RNNs. We assume that we are
    making predictions over a vocabulary of size V for each timestep of a
    timeseries of length T, over a minibatch of size N. The input x gives scores
    for all vocabulary elements at all timesteps, and y gives the indices of the
    ground-truth element at each timestep. We use a cross-entropy loss at each
    timestep, summing the loss over all timesteps and averaging across the
    minibatch.
    As an additional complication, we may want to ignore the model output at some
    timesteps, since sequences of different length may have been combined into a
    minibatch and padded with NULL tokens. The optional mask argument tells us
    which elements should contribute to the loss.
    Inputs:
    - x: Input scores, of shape (N, T, V)
    - y: Ground-truth indices, of shape (N, T) where each element is in the range
         0 <= y[i, t] < V
    - mask: Boolean array of shape (N, T) where mask[i, t] tells whether or not
      the scores at x[i, t] should contribute to the loss.
    Returns a tuple of:
    - loss: Scalar giving loss
    - dx: Gradient of loss with respect to scores x.
    """

    N, T, V = x.shape
    x = np.reshape(x, (N * T, V))
    y = np.reshape(y, (N * T))
    mask = mask.reshape(N * T)

    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    prob_x = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    prob_x2 = -np.log(prob_x[range(N*T), y])
    loss = np.sum(prob_x2 * mask)
    loss /= N

    dx = prob_x
    dx[range(N*T), y] -= 1

    mask_reshape = np.reshape(mask, (N*T, 1))
    dx = dx * mask_reshape
    dx = np.reshape(dx, ((N, T, V)))
    dx /= N
    return loss, dx