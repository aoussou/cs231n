from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from fc_net import *
from solver import Solver

# Load the IRIS data
# assign common names to the relevant variables
iris = datasets.load_iris()
X = iris.data
N = X.shape[0] # number of data points
y = iris.target
ind = np.random.permutation(N)
Xin = X[ind]
yin = y[ind]

# chose the train/validation/test set ratios
pTrain = .7
pVal = .3
# pTest = 1 - rTrain is not explicitly needed

# chose the number of hidden layers and nodes for each hidden layer
# [4, 7] means 2 hidden layers with 4 and 7 nodes in the 1st and 2nd resp.
hidden_dims = [40,60,50]
num_train = int(N*.7)
num_val = int(N*.2)
data = {
  'X_train': Xin[:num_train],
  'y_train': yin[:num_train],
  'X_val': Xin[num_train:num_train+num_val],
  'y_val': yin[num_train:num_train+num_val],
  'X_test': Xin[num_train+num_val:],
  'y_test': yin[num_train+num_val:],  
}

# Train the model for hyperparametesr
# In the interest of time and because the model is so small, here only the learning rate, regularization and dropout parameters will be trained for

# Other parameters that could have been trained for are: number of layers and their corresponding number of nodes, number of epochs 

LR = 10 ** np.random.uniform(-4, -2,size = 5)
REG = np.random.uniform(10e-4,10e-3,size = 5)
best_val_acc = 0
DO = [0, .25, .5]
for lr in LR:
    for reg in REG:
        for dp in DO:
            model = VNN(hidden_dims,input_dim=X.shape[1],dropout = dp,use_batchnorm = True,reg = reg,acti='sigmo',lossf = softmax_loss)
            
            solver = Solver(model, data,
                            num_epochs=20,
                            update_rule='adam',
                            optim_config={
                              'learning_rate': lr
                            },
                            verbose=False, print_every=1,bgd = False)
            solver.train()
    
            y_val_pred = np.argmax(model.loss(data['X_val']), axis=1)
            val_acc = (y_val_pred == data['y_val']).mean()
            if val_acc > best_val_acc:
                best_model = model
                best_val_acc = val_acc.copy()
                best_solver = solver
                best_params = [lr,reg,dp]
      

model = best_model    
solver = best_solver

print('\n \n #################################')
print('     REPORT')
print(' ################################# \n')

print('Best validation accuracy: %.2f' % (best_val_acc))        
y_test_pred = np.argmax(model.loss(data['X_test']), axis=1)
print('Test accuracy: %.2f \n' % (y_test_pred == data['y_test']).mean())
print('Best hyperparameters: \n learning rate = %.2e \n regularization = %.2e \n dropout = %.1f \n' % (best_params[0],best_params[1],best_params[2]) )

print('Please see new window for plots.　\n')

print('      m(_ _)m')

###################################

# PLOT

plt.subplot(2, 1, 1)
plt.plot(solver.loss_history, 'o')
plt.title('best model training loss')
plt.xlabel('epoch')

plt.subplot(2, 1, 2)
plt.plot(solver.val_acc_history, '-o')
plt.title('best model validation accuracy')
plt.xlabel('epoch')

plt.tight_layout()
#plt.gcf().set_size_inches(15, 15)
plt.show()
