The code is a simplified version of the cs231n framework (assignment 2, spring 2017 version) which you can find [here](http://cs231n.github.io/assignments2017/assignment2/).

The framework mostly contains files with empty functions that need to be completed, as well as a number of iPython notebooks. For the purpose of this project, irrelevant files and functions were deleted. In particular, the iPython notebooks were replaced by one Python driver file.
The svm_loss and softmax_loss function were already provided, however those who took the course already coded those functions by themselves in assignment 1.

The code is best run using Python 3, however should also work with Python 2.

FILES DESCRIPTION:

1) driver.py r: the file to be run to obtain results for the Ascent assignment with the IRIS data set
2) fc_net.py : the file that contains the Neural Network architecture. For this project only one class is used, namely VNN which sets a Neural Network with the following architecture
  {affine layer - [batch norm] - acti - [dropout]} x (L - 1) - affine - loss func
3) layer.py: a file that contains the function required to perform the relevant mathematical operations in each layer. The functions of layer.py are called in fc_net.py
4) solver.py: the file that trains the model set in fc_net.py . Calls functions in fc_net.py and uses gradient descent steps to update the parameters.
5) optim.py: a file that contains a number of function from which the update rule can be chosen. This essentially consists of gradient descent and a number of possible enhancements.

The code possesses the following features:

1) batch normalization
2) dropout
3) Softmax loss function
4) regular momentum, Adam update rule, RMSProp update rule
5) ReLU activation

The following options not originally in the cs231n code and were added:

2) sigmoid activation function (as required in the Ascent assignment)
3) SVM loss function 
4) Nesterov momentum 

NOTE: no activation was used on the output layer

Because the Iris dataset is much smaller than the CIFAR-10 used in the cs231n course, the batches passed to the stochastic gradient descent are made to have the same size as the training set, so essentially a regular gradient descent scheme is used and only one iteration per epoch is performed. Also, because the sample is small, a 70/20/10 training/validation/test set is chosen as opposed to the perhaps more traditional 60/20/20 split.

The architecture is fixed to 3 hidden layers of respectively 40, 60 and 50 nodes, however the code can handle an arbitrary number of hidden layers with an arbitrary number of nodes in each. 

This architecture was arbitrarily chosen and along with other hyperparameters shuch as the weight scales and the number of epochs/iterations, it is not trained for because it was found that if batch normalization is used, between 96.667% and 100% accuracy on the validation set is almost always achieved within 20 iterations using the Adam update rule and default value for those parameters.

The hyperparameters trained are the learning rate, the regularization and the dropout probability.

The model would be expected to perform better if the data set were larger, because the results on the TEST set are typically between 93.3333% and 100%, but sometimes fall to 86.6667% depending on the seed. The model scales arguably well with larger, more complex data sets since it can achieve above 55% accuracy on the CIFAR-10 data set, however at this level a CNN would probably be required.

