from builtins import range
import numpy as np
from past.builtins import xrange

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """

    D = int(x.size/x.shape[0])
    out = np.reshape(x,(x.shape[0],D)).dot(w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
   
    x, w, b = cache
    dx = np.reshape(dout.dot(w.T),x.shape)    
    D = int(x.size/x.shape[0])
    dw = (np.reshape(x,(x.shape[0],D)).T).dot(dout)    
    db = np.dot(( np.ones( (dout.shape[0],1) ) ).T,dout)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = x*(x>0)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    
    dx = dout*(cache>0)

    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    
    if mode == 'train':
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0)
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        xhat = (x - sample_mean)/(sample_var + eps)**.5
        out = gamma*xhat + beta       
        cache = (gamma,sample_mean,sample_var,eps,xhat,x)        

    elif mode == 'test':
        out = gamma*(x - running_mean)/(running_var + eps)**.5 + beta
        cache = None
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var


    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """

    gamma, sample_mean, sample_var, eps, xhat, x = cache
    n = xhat.shape[0]
    vareps = (sample_var+eps) 
    xcentr = x - sample_mean
    
    # NOTE: dout = dL/dz, dgamma = dL/dgamma, dbeta = dL/dbeta, dx = dL/dx
    # dL/dgamma = dz/dgamma * dL/dz
    dgamma = np.sum(xhat*dout,axis=0) #Hadmard - sum col
    
    # dL/dbeta = dz/dbeta * dL/dz
    dbeta = np.sum(dout,axis=0)       #(Hadamard w. 1s) - sum col
    
    ####### dx: IT'S GOING TO BE A TOUGHER ONE ###########
    '''
    d1 = gamma*dout
    d2 = np.sum(xcentr*d1,axis = 0)
    d3 = -np.divide(d2,vareps)
    d4 = np.divide(d3,2*vareps**.5)
    d5 = d4*np.ones(x.shape)/n
    d61 = 2*xcentr*d5
    d62 = np.divide(d1,vareps**.5)    
    d6 = d61 + d62
    d7 = -np.sum(d6,axis = 0)
    d81 = d7*np.ones(x.shape)/n
    d82 = d6
    dx = d81 + d82
    '''      
    dx =( -2*xcentr*np.divide(
            np.divide(np.sum(xcentr*(gamma*dout),axis = 0),vareps),
            2*vareps**.5)*np.ones(x.shape)/n + np.divide(gamma*dout,vareps**.5)
    -np.sum(2*xcentr*np.divide(np.divide(np.sum(xcentr*(gamma*dout),axis = 0)
    ,vareps),2*vareps**.5)*np.ones(x.shape)/n + np.divide(gamma*dout,vareps**.5)
    ,axis = 0)*np.ones(x.shape)/n)
    
    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])


    if mode == 'train':
        
        mask = (np.random.rand(*x.shape) < p) / p
        out = x*mask
        
    elif mode == 'test':
        out = np.maximum(0,x)
        mask = None
        
    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        dx = mask*dout
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    stride = conv_param['stride']; pad = conv_param['pad']
    N = x.shape[0]; H = x.shape[2]; W = x.shape[3] 
    F = w.shape[0]; HH = w.shape[2]; WW = w.shape[3]
    
    Hout = int(1 + (H + 2 * pad - HH) / stride)
    Wout = int(1 + (W + 2 * pad - WW) / stride)
    
    out = np.zeros((N,F,Hout,Wout))
    
    # npad is a tuple of (n_before, n_after) for each dimension
    npad = ((0, 0), (0, 0), (1, 1),(1,1))
    X = np.pad(x, pad_width=npad, mode='constant', constant_values=0)    
    for n in range(N):
        for f in range(F):
            for hout in range(Hout):
                for wout in range(Wout):
                    posH = stride*hout
                    posW = stride*wout
                    out[n,f,hout,wout] = np.sum(
                            X[n,:,posH:posH+HH,posW:posW+WW]*w[f,:,:,:])+ b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    
    (x, w, b, conv_param) = cache

    # npad is a tuple of (n_before, n_after) for each dimension
    npad = ((0, 0), (0, 0), (1, 1),(1,1))
    X = np.pad(x, pad_width=npad, mode='constant', constant_values=0) 
    stride = conv_param['stride']; pad = conv_param['pad']
    N = x.shape[0]; H = x.shape[2]; W = x.shape[3] 
    F = w.shape[0]; HH = w.shape[2]; WW = w.shape[3]
    
    Hout = int(1 + (H + 2 * pad - HH) / stride)
    Wout = int(1 + (W + 2 * pad - WW) / stride)    

    dw = np.zeros(w.shape)
    dx = np.zeros(X.shape)
    db = np.zeros(b.shape)

    for n in range(N):
        for f in range(F):
            for hout in range(Hout):
                for wout in range(Wout):
                    posH = stride*hout
                    posW = stride*wout                   
                    dx[n,:,posH:posH+HH,posW:posW+WW] += w[f,:,:,:] * dout                 [n,f,hout,wout]
                    dw[f,:,:,:] += X[n,:,posH:posH+HH,posW:posW+WW] * dout                 [n,f,hout,wout]
                    db[f] += np.sum(dout[n,f,hout,wout])

    dx = dx[:,:,1:-1,1:-1]

    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    N = x.shape[0]; C = x.shape[1]; H = x.shape[2]; W = x.shape[3] 
    stride = pool_param['stride']
    PH = pool_param['pool_height']   
    PW = pool_param['pool_width']   
    Hout = int((H - PH)/stride + 1)
    Wout = int((W - PW)/stride + 1)
    
    out = np.zeros((N,C,Hout,Wout))
    for n in range(N):
        for c in range(C):
            for hout in range(Hout):
                for wout in range(Wout):
                    posH = stride*hout
                    posW = stride*wout
                    out[n,c,hout,wout] = np.amax(x[n,c,posH:posH+PH,posW:posW+PW])
                    
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    
    (x, pool_param) = cache  
    N = x.shape[0]; C = x.shape[1]; H = x.shape[2]; W = x.shape[3] 
    stride = pool_param['stride']
    PH = pool_param['pool_height']   
    PW = pool_param['pool_width']   
    Hout = int((H - PH)/stride + 1)
    Wout = int((W - PW)/stride + 1)
    dx = np.zeros((N,C,H,W))
    
    tmp = np.ones((PH,PW))

    for n in range(N):
        for c in range(C):
            for hout in range(Hout):
                for wout in range(Wout):
                    posH = stride*hout
                    posW = stride*wout

                    dx0 = tmp*(x[n,c,posH:posH+PH,posW:posW+PW] - np.amax(x[n,c,posH:posH+PH,posW:posW+PW])>=0)

                    dx[n,c,posH:posH+PH,posW:posW+PW] = dx0 * dout[n,c,hout,wout]
                    
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N = x.shape[0]; C = x.shape[1]; H = x.shape[2]; W = x.shape[3]
    X = np.reshape(x,(N,C*H*W))
    GAMMA = np.repeat(gamma,H*W); BETA = np.repeat(beta,H*W)
    out, cache = batchnorm_forward(X, GAMMA, BETA, bn_param)
    
    out = np.reshape(out,x.shape)

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """ 
    N = dout.shape[0]; C = dout.shape[1]; H = dout.shape[2]; W = dout.shape[3]     
  
    DOUT = np.reshape(dout,(N,C*H*W))
    dx, dgamma, dbeta = batchnorm_backward(DOUT, cache)
    
    dx = np.reshape(dx,dout.shape)
    dgamma = np.sum(np.reshape(dgamma,(C,H*W)),axis=1)
    dbeta = np.sum(np.reshape(dbeta,(C,H*W)),axis=1)

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx
