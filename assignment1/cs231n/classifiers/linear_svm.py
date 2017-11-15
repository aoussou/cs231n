import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]    
    for j in xrange(num_classes):
      if j == y[i]:
        #print(scores[j] - correct_class_score + 1)
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i].T #dWyi
        dW[:,j] += X[i].T    #dWj

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  scores = np.dot(X,W) # size (N,C)
  correct_class_score = scores[np.arange(scores.shape[0]),y]

  margin = (scores.T - correct_class_score + 1).T # size (N,C)
  margin[np.arange(margin.shape[0]),y] = 0 
  ind0 = np.where(margin<=0)
  margin[ind0] = 0

  loss = np.sum(margin)/X.shape[0] + reg * np.sum(W * W)  
 ######################################################## 
 
 #dWj
  M = np.zeros(margin.shape)  
  M[np.where(margin>0)] = 1
  dW = np.dot(X.T,M)
  
  #dWyi
  Ms = np.sum(M,1)
  Xs = X.T*Ms
  Y0 = np.zeros(margin.shape)
  Y0[np.arange(scores.shape[0]),y] = 1
  dWy = np.dot(Xs,Y0)
  dW -=  dWy
  dW = dW/X.shape[0] + 2*reg*W  

  return loss, dW
