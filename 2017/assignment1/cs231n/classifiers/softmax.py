import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros(W.shape)

  for i in xrange(X.shape[0]):
      
#      f = np.dot(X[i],W)
#      fn = np.exp(f - np.max(f))
#      S = fn[y[i]]/sum(fn)
#      loss += -np.log(S)

      f = np.dot(X[i],W)
      fn = f - np.max(f)
      sumj = sum(np.exp(fn))

      loss += -fn[y[i]] + np.log(sumj)              
      
      for j in xrange(W.shape[1]):
          S = np.exp(fn[j])/sumj
          if j == y[i]:
              dW[:,j] += X[i]*(S-1.0)          
          else:
              dW[:,j] += X[i]*S

  loss /= X.shape[0]
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW/X.shape[0] + 2*reg*W

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
#  F = np.dot(X,W) # N x C
#  Fn = np.exp(F.T - np.max(F,1)).T  
#  sumj = np.sum(Fn,1)
#  S = (Fn.T/sumj).T  
#  S0 = S[np.arange(Fn.shape[0]),y]
#  loss = sum( -np.log(S0))/X.shape[0] + reg * np.sum(W * W) 
  
  #######################################
  F = np.dot(X,W) # N x C
  Fn = (F.T - np.max(F,1)).T  
  sumj = np.sum(np.exp(Fn),1)
  loss = sum(-Fn[np.arange(Fn.shape[0]),y] + np.log(sumj))/X.shape[0] + reg * np.sum(W * W)   
  #######################################

  dLdF= (np.exp(Fn).T/sumj).T # = REGULAR SOFTMAX 
  
  # DERIVATIVE OF SM LOSS wrt what's inside the exponential
  dLdF[np.arange(dLdF.shape[0]),y] -= 1.0 
  dW = np.dot(X.T,dLdF)
  
  #####!!! CAUTION !!!#######
  # if you directly plug in dLdF, without the -1.0, in some cases it work but
  # it is WRONG
  ##############  
  
  
  # USING A MASK
#  S = (np.exp(Fn).T/sumj).T   
#  #dWj
#  Xsj = np.dot(X.T,S)
#  dW +=  Xsj  
#  
#  #dWyi
#  S0 = S[np.arange(Fn.shape[0]),y]  
#  Xsy = -(X.T * np.ones(S0.shape)).T
#  Ysy = np.zeros(F.shape)
#  Ysy[np.arange(Ysy.shape[0]),y] = 1
#  dWy = np.dot(Xsy.T,Ysy)
#  dW +=  dWy  
  
  
  dW = dW/X.shape[0] + 2*reg*W  


  return loss, dW

