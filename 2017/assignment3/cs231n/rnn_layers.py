from __future__ import print_function, division
from builtins import range
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
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)
    cache = (x,prev_h, Wx, Wh, b, next_h)
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
    x, prev_h, Wx, Wh, b, next_h = cache    
    dtanh = dnext_h*(1 - next_h**2)

    dWx = x.T.dot(dtanh)
    dWh = prev_h.T.dot(dtanh)
    dx = dtanh.dot(Wx.T)
    dprev_h = dtanh.dot(Wh.T)
    db = sum(dtanh)
    ##############################################################################
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    """
    Run a vanilla RNN forward on an entire sequence of data. We assume an input
    sequence composed of T vectors, each of dimension D. The RNN uses a hidden
    size of H, and we work over a minibatch containing N sequences. After running
    the RNN forward, we return the hidden states for all timesteps.

    Inputs:
    - x: Input data for the entire timeseries, of shape (N, T, D).
    - h0: Initial hidden state, of shape (N, H)
    - Wx: Weight matrix for input-to-hidden connections, of shape (D, H)
    - Wh: Weight matrix for hidden-to-hidden connections, of shape (H, H)
    - b: Biases of shape (H,)

    Returns a tuple of:
    - h: Hidden states for the entire timeseries, of shape (N, T, H).
    - cache: Values needed in the backward pass
    """
    ##############################################################################
    T = x.shape[1]
    h = np.zeros( (x.shape[0],T,h0.shape[1]))
    prev_h = h0
    cache = {}
    for t in range(T) :
      prev_h, cache_t = rnn_step_forward(x[:,t,:], prev_h, Wx, Wh, b)
      h[:,t,:] = prev_h
      cache[t] = cache_t
  
    cache['D'] = x.shape[2]
    ##############################################################################
    return h, cache


def rnn_backward(dh, cache):
    """
    Compute the backward pass for a vanilla RNN over an entire sequence of data.

    Inputs:
    - dh: Upstream gradients of all hidden states, of shape (N, T, H)

    Returns a tuple of:
    - dx: Gradient of inputs, of shape (N, T, D)
    - dh0: Gradient of initial hidden state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, H)
    - db: Gradient of biases, of shape (H,)
    """
    dx, dh0, dWx, dWh, db = 0, 0, 0, 0, 0
    ##############################################################################
    T = dh.shape[1]
    dx = np.zeros( (dh.shape[0],T,cache['D']) )
    dh0 = 0
    for t in range(T-1,-1,-1):
      dxt, dprev_ht, dWxt, dWht, dbt = rnn_step_backward(dh[:,t,:] + dh0, cache[t])
      dh0 = dprev_ht 
      dx[:,t,:] = dxt
      dWx += dWxt
      dWh += dWht
      db += dbt   
    ##############################################################################
    return dx, dh0, dWx, dWh, db


def word_embedding_forward(x, W):
    """
    Forward pass for word embeddings. We operate on minibatches of size N where
    each sequence has length T. We assume a vocabulary of V words, assigning each
    to a vector of dimension D.

    Inputs:
    - x: Integer array of shape (N, T) giving indices of words. Each element idx
      of x muxt be in the range 0 <= idx < V.
    - W: Weight matrix of shape (V, D) giving word vectors for all words.

    Returns a tuple of:
    - out: Array of shape (N, T, D) giving word vectors for all input words.
    - cache: Values needed for the backward pass
    """
    ##############################################################################
    
    out = W[x]    
    
    cache = x, W.shape[0]

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

    ##############################################################################
    x, V = cache
    dW = np.zeros((V,dout.shape[2]))
    np.add.at(dW,x,dout)
    ##############################################################################
    return dW


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
    
    #############################################################################
    H = prev_h.shape[1]
    
    # activation vector
    a = x.dot(Wx) + prev_h.dot(Wh) + b
    ai = a[:,:H]
    af = a[:,H:2*H]
    ao = a[:,2*H:3*H]
    ag = a[:,3*H:]
    
    i = sigmoid(ai)
    f = sigmoid(af)
    o = sigmoid(ao)
    g = np.tanh(ag)
    
    next_c = f*prev_c + i*g
    next_h = o*np.tanh(next_c)

    cache = x, prev_h, prev_c, Wx, Wh, next_c, prev_c, i, f, o, g
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
    #############################################################################
    H = dnext_h.shape[1]

    x, prev_h, prev_c, Wx, Wh, next_c, prev_c, i, f, o, g = cache
    N, D = x.shape
    
    Wxi = Wx[:,:H]
    Wxf = Wx[:,H:2*H]
    Wxo = Wx[:,2*H:3*H]
    Wxg = Wx[:,3*H:]    
    
    Whi = Wh[:,:H]
    Whf = Wh[:,H:2*H]
    Who = Wh[:,2*H:3*H]
    Whg = Wh[:,3*H:]  
 
    tmp0 = dnext_h*o*(1 - np.tanh(next_c)**2)     
    ######################################
    
    # dx
    dcdx = (dnext_c*prev_c*f*(1-f)).dot(Wxf.T) + (dnext_c*i*(1-g**2)).dot(Wxg.T) + (dnext_c*g*i*(1-i)).dot(Wxi.T)
    

    tmp1 = (dnext_h*np.tanh(next_c)*o*(1-o)).dot(Wxo.T)     
    tmp2 = (tmp0*prev_c*f*(1-f)).dot(Wxf.T) + (tmp0*i*(1-g**2)).dot(Wxg.T) + (tmp0*g*i*(1-i)).dot(Wxi.T)

    dx = dcdx + tmp1 + tmp2 
    
    # dprev_h
    dcdh = (dnext_c*prev_c*f*(1-f)).dot(Whf.T) + (dnext_c*i*(1-g**2)).dot(Whg.T) + (dnext_c*g*i*(1-i)).dot(Whi.T)
    
    tmp1 = (dnext_h*np.tanh(next_c)*o*(1-o)).dot(Who.T)     
    tmp2 = (tmp0*prev_c*f*(1-f)).dot(Whf.T) + (tmp0*i*(1-g**2)).dot(Whg.T) + (tmp0*g*i*(1-i)).dot(Whi.T)
        
    dprev_h = dcdh + tmp1 + tmp2
    
    # dprev_c
    dprev_c = dnext_c*f + f*tmp0
     
    #### dWx    
    # dWxi
    dcdwxi = x.T.dot(dnext_c*g*i*(1-i)) 
    dhdwxi = x.T.dot(tmp0*g*i*(1-i))    
    dWxi = dcdwxi + dhdwxi

    # dWxf
    dcdwxf = x.T.dot(dnext_c*prev_c*f*(1-f))
    dhdwxf = x.T.dot(tmp0*prev_c*f*(1-f))    
    dWxf = dcdwxf + dhdwxf

    # dWxo
    dWxo = x.T.dot(dnext_h*np.tanh(next_c)*o*(1-o))

    # dWxg
    dcdwxg = x.T.dot(dnext_c*i*(1 - g**2))    
    dhdwxg = x.T.dot(tmp0*i*(1 - g**2))    
    dWxg = dcdwxg + dhdwxg

    dWx = np.concatenate((dWxi,dWxf,dWxo,dWxg),axis=1) 

    #### dWx    
    # dWxi
    dcdwhi = prev_h.T.dot(dnext_c*g*i*(1-i)) 
    dhdwhi = prev_h.T.dot(tmp0*g*i*(1-i))    
    dWhi = dcdwhi + dhdwhi

    # dWxf
    dcdwhf = prev_h.T.dot(dnext_c*prev_c*f*(1-f))
    dhdwhf = prev_h.T.dot(tmp0*prev_c*f*(1-f))    
    dWhf = dcdwhf + dhdwhf

    # dWxo
    dWho = prev_h.T.dot(dnext_h*np.tanh(next_c)*o*(1-o))

    # dWxg
    dcdwhg = prev_h.T.dot(dnext_c*i*(1 - g**2))    
    dhdwhg = prev_h.T.dot(tmp0*i*(1 - g**2))    
    dWhg = dcdwhg + dhdwhg

    dWh = np.concatenate((dWhi,dWhf,dWho,dWhg),axis=1) 
    
    ############################
    # dbi
    dcdbi = np.sum(dnext_c*g*i*(1-i),axis=0) 
    dhdbi = np.sum(tmp0*g*i*(1-i),axis=0)    
    dbi = dcdbi + dhdbi

    # dbf
    dcdbf = np.sum(dnext_c*prev_c*f*(1-f),axis=0)
    dhdbf = np.sum(tmp0*prev_c*f*(1-f),axis=0)    
    dbf = dcdbf + dhdbf

    # dWxo
    dbo = np.sum(dnext_h*np.tanh(next_c)*o*(1-o),axis=0)

    # dWxg
    dcdbg = np.sum(dnext_c*i*(1 - g**2),axis=0)    
    dhdbg = np.sum(tmp0*i*(1 - g**2),axis=0)    
    dbg = dcdbg + dhdbg
 
    db = np.concatenate((dbi,dbf,dbo,dbg),axis=0)
  
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
    - x: Input data of shape (N, T, D)
    - h0: Initial hidden state of shape (N, H)
    - Wx: Weights for input-to-hidden connections, of shape (D, 4H)
    - Wh: Weights for hidden-to-hidden connections, of shape (H, 4H)
    - b: Biases of shape (4H,)

    Returns a tuple of:
    - h: Hidden states for all timesteps of all sequences, of shape (N, T, H)
    - cache: Values needed for the backward pass.
    """    
    #############################################################################
    N = x.shape[0]
    T = x.shape[1]
    H = h0.shape[1]
    h = np.zeros((N,T,H)) 
    h_prev = h0
    c = np.zeros_like(h0)
    cache = {}
    for t in range(T):
      h_prev, c, cache_t = lstm_step_forward(x[:,t,:], h_prev, c, Wx, Wh, b)
      h[:,t,:] = h_prev
      cache[t] = cache_t

    cache['D'] = x.shape[2]
    ##############################################################################

    return h, cache


def lstm_backward(dh, cache):
    """
    Backward pass for an LSTM over an entire sequence of data.]

    Inputs:
    - dh: Upstream gradients of hidden states, of shape (N, T, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data of shape (N, T, D)
    - dh0: Gradient of initial hidden state of shape (N, H)
    - dWx: Gradient of input-to-hidden weight matrix of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weight matrix of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dh0, dWx, dWh, db = 0, 0, 0, 0, 0
    #############################################################################
    T = dh.shape[1]
    dx = np.zeros( (dh.shape[0],T,cache['D']) )
    dprev_ct = 0
    for t in range(T-1,-1,-1):
      dxt, dprev_ht, dprev_ct, dWxt, dWht, dbt = lstm_step_backward(dh[:,t,:] + dh0, dprev_ct, cache[t])

      dh0 = dprev_ht 
      dx[:,t,:] = dxt
      dWx += dWxt
      dWh += dWht
      db += dbt         
      
    ##############################################################################

    return dx, dh0, dWx, dWh, db


def temporal_affine_forward(x, w, b):
    """
    Forward pass for a temporal affine layer. The input is a set of D-dimensional
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
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


def temporal_affine_backward(dout, cache):
    """
    Backward pass for temporal affine layer.

    Input:
    - dout: Upstream gradients of shape (N, T, M)
    - cache: Values from forward pass

    Returns a tuple of:
    - dx: Gradient of input, of shape (N, T, D)
    - dw: Gradient of weights, of shape (D, M)
    - db: Gradient of biases, of shape (M,)
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


def temporal_softmax_loss(x, y, mask, verbose=False):
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

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1
    dx_flat /= N
    dx_flat *= mask_flat[:, None]

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx
