from numpy.core.numeric import NaN
import funcs
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import NeuralNetwork
import activation as act
import costfuncs as cost
import tensorflow as tf
from NN_kreas import Keras
import seaborn as sb
from analysis_funcs import ols, test_activation, prep_network_franke, train_network_reg


'''
This is a testing file. Run this file to runs the tests done on the the franke data with Neural Network.
To run all the tests the variable run_all must be set to True.  

The file contains the following functions:
    plot_heatmap: Plots heatmap illustrating the results 
    test_epochs_bsize: Testing different values for epochs and batch size
    test_bsize_hidden: Testing different values for batch size and number of hidden layers
    test_eta_lmd: Testing different combinations of eta and lambda. 
'''


np.random.seed(883)

#Set to True if you want to run all the testing
run_all = False

# Creates data from FrankeFunction
# This data is used in all the analysis in this file
x, y, z = funcs.make_data(20)

# The finding the best possible mse with OLS
mse_ols = ols(x,y,z)
print("MSE OLS: ", mse_ols)

#train and test data for the neural network
X_train, X_test, z_train, z_test = prep_network_franke(x,y,z)

bsize = 32
n_epochs = 300
eta = 1e-3
n_hidden = [50]

test_activation(X_train, X_test, z_train, z_test,
                bsize, n_epochs, eta, n_hidden, 'figs_NN_franke/one_hidden50')


bsize = 32
n_epochs = 300
eta = 1e-3
n_hidden = [20,20]

test_activation(X_train, X_test, z_train, z_test,
                bsize, n_epochs, eta, n_hidden, 'figs_NN_franke/two_hidden20')



bsize = 32
n_epochs = 500
eta = 1e-3
n_hidden = [10,10,10]

test_activation(X_train, X_test, z_train, z_test,
                bsize, n_epochs, eta, n_hidden, 'figs_NN_franke/three_hidden_2')



#Comparing sigmoid and relu with keras:
NN_keras = Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, 0, loss='mean_squared_error',
                 metrics=[], act_hidden='sigmoid', act_out=tf.identity)

model = NN_keras.NN_model()

model.fit(X_train, z_train, epochs=n_epochs, batch_size=bsize, verbose=0)
scores = model.evaluate(X_test, z_test)
print(f"MSE test keras (sigmoid) {scores}")


NN_keras = Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, 0, loss='mean_squared_error',
                  metrics=[], act_hidden='relu', act_out=tf.identity)

model = NN_keras.NN_model()

model.fit(X_train, z_train, epochs=n_epochs, batch_size=bsize, verbose=0)
scores = model.evaluate(X_test, z_test)
print(f"MSE test keras (relu) {scores}")



def plot_heatmap(MSE, xval, yval, xlab, ylab, title, filename=None):
    heatmap = sb.heatmap(MSE, annot=True, fmt='.4g', cmap='YlGnBu',
                     xticklabels=xval, yticklabels=yval,
                     cbar_kws={'label': 'MSE'})
    heatmap.set_ylabel(ylab, size=12)
    heatmap.set_xlabel(xlab, size=12)
    heatmap.invert_yaxis()
    heatmap.set_title(title, size=16)
    if filename:
        plt.savefig('figs_NN_franke/' + filename)


def test_epochs_bsize(act_hidden, filename):

    n_in = X_train.shape[1]
    n_out = z_train.shape[1]
    n_hidden = [50]
    act_out = act.Activation_Identity()

    gamma = 0
    eta = 1e-3

    my_cost = cost.MSE()

    epochs = range(50,501, 50)
    bsizes = [8,16,32,64,128]
    MSEs = np.zeros((len(bsizes), len(epochs)))


    if isinstance(act_hidden, act.Activation_Sigmoid):
        title = "MSE of Sigmoid + Identity, epochs vs batch size"
    elif isinstance(act_hidden, act.Activation_ReLU):
        title = "MSE of ReLU + Identity, epochs vs batch size"
    elif isinstance(act_hidden, act.Activation_LeakyReLU):
        title = "MSE of Leaky ReLU + Identity, epochs vs batch size"


    plt.figure(figsize = [12, 8])

    print('Testing epochs vs bsize for NN with franke ....')
    for i in range(len(epochs)):
        print(i+1, '/', len(epochs))
        for j in range(len(bsizes)):

            network = NeuralNetwork(n_in, n_out, act_hidden, act_out, my_cost,
                                    n_in_each_hidden=n_hidden)
            layers = network.create_layers()

            mse_train, mse_test = train_network_reg(network, bsizes[j], epochs[i], eta,
                                                    X_train, X_test, z_train, z_test)
            
            #Avoid high values in the plot to be able to distinguish small values by color in the heatmap
            if mse_test[-1] < 1:
                MSEs[j,i] = mse_test[-1]
            else:
                MSEs[j,i] = NaN


    plot_heatmap(MSEs, epochs, bsizes, "Epochs", 'Size of minibatches', title, filename=filename)
    plt.show()


def test_bsize_hidden(act_hidden, filename):


    if isinstance(act_hidden, act.Activation_Sigmoid):
        title = "MSE of Sigmoid + Identity, hidden layers vs batch size"
    elif isinstance(act_hidden, act.Activation_ReLU):
        title = "MSE of ReLU + Identity, hidden layers vs batch size"
    elif isinstance(act_hidden, act.Activation_LeakyReLU):
        title = "MSE of Leaky ReLU + Identity, hidden layers vs batch size"

    n_in = X_train.shape[1]
    n_out = z_train.shape[1]
    act_out = act.Activation_Identity()
    gamma = 0
    eta = 1e-3
    my_cost = cost.MSE()
    epochs = 300

    n_hiddens = [[10], [50], [100], [10,10], [30,30], [50,50], [10,10,10],[15,15,15],[20,20,20]]
    bsizes = [8,16,32,64,128]
    MSEs = np.zeros((len(n_hiddens), len(bsizes)))
    plt.figure(figsize = [12, 8])

    print('Testing hidden layers vs bsize for NN with franke ....')
    for i in range(len(n_hiddens)):
        print(i+1, '/', len(n_hiddens))
        for j in range(len(bsizes)):

            network = NeuralNetwork(n_in, n_out, act_hidden, act_out, my_cost,
                                    n_in_each_hidden=n_hiddens[i])
            layers = network.create_layers()

            mse_train, mse_test = train_network_reg(network, bsizes[j], epochs, eta,
                                                    X_train, X_test, z_train, z_test)
            if mse_test[-1] < 1:
                MSEs[i,j] = mse_test[-1]
            else: 
                MSEs[i,j] = NaN 


    plot_heatmap(MSEs, bsizes, n_hiddens, 'Size of minibatches', 'n_hidden', title, filename=filename)
    plt.show()


def test_eta_lmd(act_hidden, filename):
    n_in = X_train.shape[1]
    n_out = z_train.shape[1]
    n_hidden = [50]
    act_out = act.Activation_Identity()
    epochs = 300
    bsize = 16

    lmds = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    etas = [1e-4, 1e-3, 1e-2, 0.02, 0.03]

    my_cost = cost.MSE()

    MSEs = np.zeros((len(etas), len(lmds)))

    if isinstance(act_hidden, act.Activation_Sigmoid):
        title = "MSE of Sigmoid + Identity, eta vs lmd"
    elif isinstance(act_hidden, act.Activation_ReLU):
        title = "MSE of ReLU + Identity, eta vs lmd"
    elif isinstance(act_hidden, act.Activation_LeakyReLU):
        title = "MSE of Leaky ReLU + Identity, eta vs lmd"

    plt.figure(figsize = [12, 8])

    print('Testing eta vs lmd for NN with franke ....')
    for i in range(len(etas)):
        print(i+1, '/', len(etas))
        for j in range(len(lmds)):

            network = NeuralNetwork(n_in, n_out, act_hidden, act_out, my_cost,
                                    n_in_each_hidden=n_hidden)
            layers = network.create_layers()

            mse_train, mse_test = train_network_reg(network, bsize, epochs, etas[i],
                                                    X_train, X_test, z_train, z_test, lmd=lmds[j])

            if mse_test[-1] < 1:
                MSEs[i,j] = mse_test[-1]
            else: 
                MSEs[i,j] = NaN 


    plot_heatmap(MSEs, lmds, etas, 'lambdas', 'Learning rate', title, filename=filename)
    plt.show()



if run_all:
    # Analysing epochs and batch size 
    test_epochs_bsize(act.Activation_Sigmoid(), 'ep_b_sig')
    test_epochs_bsize(act.Activation_ReLU(), 'ep_b_relu')
    test_epochs_bsize(act.Activation_LeakyReLU(), 'ep_b_lrelu')

    # Analysing hidden layers and batch size 
    test_bsize_hidden(act.Activation_Sigmoid(), 'hid_b_sig')
    test_bsize_hidden(act.Activation_ReLU(), 'hid_b_relu')
    test_bsize_hidden(act.Activation_LeakyReLU(), 'hid_b_lrelu')

    # Analysing eta and lambda 
    test_eta_lmd(act.Activation_Sigmoid(), 'eta_lmd_sig')
    test_eta_lmd(act.Activation_ReLU(), 'eta_lmd_relu')
    test_eta_lmd(act.Activation_LeakyReLU(), 'eta_lmd_lrelu')
