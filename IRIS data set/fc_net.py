from builtins import range
from builtins import object
import numpy as np
from layers import *



class VNN(object):
    """
    A fully-connected neural network with an arbitrary number of hidden
    layers, a sigmoid or relu activation layers after hidden layer, no activation 
    after the output layer. 
    
    This will also implement dropout and batch normalization as options. For a 
    network with L layers, the architecture will be

    {affine - [batch norm] - acti - [dropout]} x (L - 1) - affine - loss func

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None,acti = 'relu',lossf = softmax_loss):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input. # i.e D of (N,D)
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        if acti == 'relu':
            self.activation_forward = relu_forward
            self.activation_backward = relu_backward    
        elif acti == 'sigmo':
            self.activation_forward = sigmoid_forward
            self.activation_backward = sigmoid_backward  
            
        self.loss_function = lossf
         
        #1st LAYER
        self.params['W1'] = weight_scale*np.random.randn(input_dim,hidden_dims[0])
        self.params['b1'] = np.zeros(hidden_dims[0])      
        
        for i in range(0,len(hidden_dims)-1):
            self.params['W'+str(i+2)] = weight_scale*np.random.randn(
                    hidden_dims[i],hidden_dims[i+1])   
            self.params['b'+str(i+2)] = np.zeros(hidden_dims[i+1])  
            
            if self.use_batchnorm:
                self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
                self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i])

        self.params['W'+str(self.num_layers)] = weight_scale*np.random.randn(
                hidden_dims[-1],num_classes)        
        self.params['b'+str(self.num_layers)] = np.zeros(num_classes)
 
        if self.use_batchnorm:
            self.params['gamma'+str(self.num_layers-1)] = (
                    np.ones(hidden_dims[-1]))
            self.params['beta'+str(self.num_layers-1)] = (
                    np.zeros(hidden_dims[-1]))

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        #print(X)
        #print('\n after copy best W100',self.params['W1'][0,0])        
        
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        grads = {}
        Xtmp = X;
        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
            
        REG = 0
        CACHE = {}
        CACHE_BN = {}
        CACHE_DO = {}
        
        for i in range(0,self.num_layers-1):

             Xtmp, cache = affine_forward(Xtmp, self.params['W'+str(i+1)], self.params['b'+str(i+1)])                 
             CACHE['cache'+str(i+1)] = cache
             
             if self.use_dropout:
                 Xtmp, cache = dropout_forward(Xtmp,self.dropout_param)
                 CACHE_DO['cache'+str(i+1)] = cache  
             if self.use_batchnorm:                        
                 Xtmp, cache = batchnorm_forward(
                         Xtmp, 
                         self.params['gamma'+str(i+1)], 
                         self.params['beta'+str(i+1)],
                         self.bn_params[i])
                 CACHE_BN['cache'+str(i+1)] = cache  
                            
             Xtmp = self.activation_forward(Xtmp)[0]
             REG += np.sum(self.params['W'+str(i+1)] * self.params['W'+str(i+1)])                 
      

        
        # OUTPUT FROM LAST LAYER (i.e. SCORES) NOT NORMALIZED                
        scores, cacheL = affine_forward(Xtmp,
                                         self.params['W'+str(self.num_layers)],
                                         self.params['b'+str(self.num_layers)])  


        REG += np.sum(self.params['W'+str(self.num_layers)] * self.params['W'+str(self.num_layers)])        


        # If test mode return early
        if mode == 'test':
            return scores

        loss, dlossdx = self.loss_function(scores, y) 
        loss = loss + .5*self.reg * REG
        
        dx, dw, db = affine_backward(dlossdx, cacheL)
      

        grads['W'+str(self.num_layers)] = dw + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = np.reshape(db.T,(db.size,))

        da = self.activation_backward(dx, cacheL[0])         
        for i in range(self.num_layers-1,0,-1):
            if self.use_batchnorm:                                 
                da,grads['gamma'+str(i)],grads['beta'+str(i)] = batchnorm_backward(da, CACHE_BN['cache'+str(i)])
                
            if self.use_dropout:
                da  = dropout_backward(da, CACHE_DO['cache'+str(i)])
            
            dx, dw, db = affine_backward(da, CACHE['cache'+str(i)])                        
            grads['W'+str(i)] = dw + self.reg*self.params['W'+str(i)]
            grads['b'+str(i)] = np.reshape(db.T,(db.size,))
            da = self.activation_backward(dx, CACHE['cache'+str(i)][0])       

        return loss, grads
