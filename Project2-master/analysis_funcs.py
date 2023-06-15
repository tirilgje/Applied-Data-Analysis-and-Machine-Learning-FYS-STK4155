
import funcs
import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import NeuralNetwork
import activation as act
import costfuncs as cost
import pandas
from sklearn.datasets import load_breast_cancer


'''
This file contains function used in the analysis of the neural network and logistc regression
for both franke and breact canser data
The logistic regression code is implemented in the neural network class.

Contains the following functions:
    ols: calculates the OLS score with the data 
    plot_results: Plots the data against epochs and optionally saves the figure 
    prep_network_bc: reads the breast cancer data, returns scaled test and train data
    prep_network_franke: preps the franke data, returns scaled test and train data 
    train_network_class: trains a network using backprop on classification problem
    train_network_class: trains a network using backprop on regression problem
    run_network_class: sets all the parameters and runs the network
    test_activation: compares the three activation function used in hidden layer in the same plot. 
'''

def ols(x,y,z):
    '''
    Create data and calculate MSE with OLS regression

    Arguments:
        x,y (matrix): input values
        z (matrix): output values
    '''
    # Create designmatrix
    X = funcs.create_X(x, y, n=5)

    # splits and scales the data
    data = funcs.prep_regression(X, z)

    X_train = data["X_train_scaled"]
    X_test = data["X_test_scaled"]
    z_train = data["z_train_scaled"]
    z_test = data["z_test_scaled"]

    # Finds optimal beta and the corresponding MSE value
    beta = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ z_train)
    z_tilde_ols = X_test @ beta
    mse_ols = funcs.MSE(z_tilde_ols, z_test)

    return mse_ols

def plot_results(vals, label, xlab, ylab, title, ylim=[0.7,1], filename=None):
    '''
    Plots test and train mse against epoch
    '''
    plt.plot(vals, label = label)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.ylim(ylim)
    plt.legend()
    if filename:
        plt.savefig(filename)

def prep_network_bc():
    '''
    Loads and prepare the breast cancer data for analysis
    '''

    cancer=load_breast_cancer()      #Download breast cancer dataset

    inputs=cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
    outputs=cancer.target                  #Label array of 569 rows (0 for benign and 1 for malignant)
    labels=cancer.feature_names[0:30]
    outputs = outputs.reshape(outputs.shape[0],1)

    #splits and scales the data with standard scaler
    data = funcs.prep_regression(inputs, outputs)

    X_train = data["X_train_scaled"]
    X_test = data["X_test_scaled"]
    z_train = data["z_train_scaled"]
    z_test = data["z_test_scaled"]

    # in this case we dont want to scale the output of 1 and 0.
    z_train = np.where(z_train > 0.5, 1, 0)
    z_test = np.where(z_test > 0.5, 1, 0)

    return X_train, X_test, z_train, z_test

def prep_network_franke(x,y,z):
    '''
    Create scaled test and train data for the neural network analysis
    '''
    inputs = np.stack((x,y), axis=1)
    z = z.reshape(z.shape[0], 1)    # reshapes to avoid problems with z

    data = funcs.prep_regression(inputs, z)

    #splitting and scaling the x and y values
    X_train = data["X_train_scaled"]
    X_test = data["X_test_scaled"]
    z_train = data["z_train_scaled"]
    z_test = data["z_test_scaled"]

    return X_train, X_test, z_train, z_test

def train_network_class(network, batch_size, n_epochs, eta, X_train, X_test, z_train, z_test):
    '''
    Takes a network and parameter inputs as input and train the nettwork as a classification problem
    Can be applied to both neural network and logisitc regression classification. 
    Returns the accuracy scores for both test and train. 
    '''

    minibatches = int(z_train.shape[0]/batch_size)

    # train the network using minibatches
    training_accs = []
    test_accs = []

    index = np.arange(len(X_train))
    for k in range(n_epochs):
        np.random.shuffle(index)
        X_batches = np.array_split(X_train[index], minibatches)
        z_batches = np.array_split(z_train[index], minibatches)

        for l in range(minibatches):
            r = np.random.randint(0, minibatches-1)
            #print('til backprop', X_batches[l].shape, z_batches[l].shape)
            if network.n_in_each_hidden:

                network.backprop(X_batches[r], z_batches[r], eta=eta)
            else:
                network.logistic(X_batches[r], z_batches[r], eta=eta)


        z_tilde_train = network.feed_forward(X_train)
        z_tilde_test = network.feed_forward(X_test)
        z_tilde_test = np.where(z_tilde_test > 0.5, 1, 0)
        z_tilde_train = np.where(z_tilde_train > 0.5, 1, 0)

        acc_train = funcs.Accuracy(z_tilde_train, z_train)
        acc_test = funcs.Accuracy(z_tilde_test, z_test)

        training_accs.append(acc_train)
        test_accs.append(acc_test)

        #print(k, np.mean(network2.layers[0].bias), np.mean(network2.layers[0].weight))
        #print(k, np.mean(network2.layers[1].bias), np.mean(network2.layers[1].weight))

    return training_accs, test_accs

def train_network_reg(network, batch_size, n_epochs, eta, X_train, X_test, z_train, z_test, lmd=0):

    '''
    Takes a network and parameter inputs as input and train the nettwork as a regression problem
    Returns the accuracy scores for both test and train. 
    '''

    minibatches = int(z_train.shape[0]/batch_size)

    # train the network using minibatches
    training_mse_values_test = []
    training_mse_values_train = []

    index = np.arange(len(X_train))
    for k in range(n_epochs):
        np.random.shuffle(index)
        X_batches = np.array_split(X_train[index], minibatches)
        z_batches = np.array_split(z_train[index], minibatches)

        for l in range(minibatches):
            r = np.random.randint(0, minibatches)
            #print('til backprop', X_batches[l].shape, z_batches[l].shape)
            network.backprop(X_batches[r], z_batches[r], eta=eta, lmd=lmd)

        z_tilde_train = network.feed_forward(X_train)
        z_tilde_test = network.feed_forward(X_test)

        mse1 = funcs.MSE(z_tilde_train, z_train)
        mse2 = funcs.MSE(z_tilde_test, z_test)

        training_mse_values_test.append(mse2)
        training_mse_values_train.append(mse1)

        #print(k, np.mean(network2.layers[0].bias), np.mean(network2.layers[0].weight))
        #print(k, np.mean(network2.layers[1].bias), np.mean(network2.layers[1].weight))

    return training_mse_values_train, training_mse_values_test

def run_network_class(X_train, X_test, z_train, z_test, bsize, n_epochs, eta, act_hidden, act_out, my_cost, n_hidden, lmd=0):

    '''
    Creates a network and calls on trains_network_class to train it. 
    Returns both training and testing accuracy score. 
    '''

    n_in = X_train.shape[1]                        # = 30 (features)
    n_out = z_train.shape[1]                       # 1 (1 or 0)

    network = NeuralNetwork(n_in, n_out, act_hidden, act_out, my_cost, n_in_each_hidden=n_hidden)
    layers = network.create_layers()

    #Testing MSE before training the network
    z_tilde = network.feed_forward(X_test)

    predictions = np.where(z_tilde > 0.5, 1, 0)
    #print('Accuracy before training:', funcs.Accuracy(predictions, z_test))

    accs_train, accs_test = train_network_class(network, bsize, n_epochs, eta, X_train, X_test, z_train, z_test)


    # Runs trained neural network with testing data
    z_tilde_train = network.feed_forward(X_train)
    z_tilde_test = network.feed_forward(X_test)

    pred_test = np.where(z_tilde_test > 0.5, 1, 0)
    pred_train = np.where(z_tilde_train > 0.5, 1, 0)

    #print(f"Accuracy after training train: {funcs.Accuracy(pred_train, z_train)}")
    #print(f"Accuracy after training test: {funcs.Accuracy(pred_test, z_test)}")


    return accs_train, accs_test

def test_activation(X_train, X_test, z_train, z_test, bsize, n_epochs, eta, n_hidden, filename):
    '''
    Testting Sigmoid, ReLU and Leaky ReLU as activation functions in the hidden layers in a network
    with various number of hidden layers and number of nodes in each layer
    '''

    n_in = X_train.shape[1]                 # = 2 (x,y)
    n_out = z_train.shape[1]                # = 1 (z)
    act_out = act.Activation_Identity()     # using identity for regression problems
    my_cost = cost.MSE()                    # using MSE for regression problems
    #  Varying values
    act_funcs = [act.Activation_Sigmoid(), act.Activation_ReLU(), act.Activation_LeakyReLU()]
    name_act = ['Sigmoid', 'ReLU', 'Leaky ReLU']

    #testing with one layer
    for i in range(len(act_funcs)):

        network = NeuralNetwork(n_in, n_out, act_funcs[i], act_out, my_cost, n_in_each_hidden=n_hidden)
        layers = network.create_layers()

        #Testing MSE before training the network
        z_tilde = network.feed_forward(X_test)
        print('MSE before training (' + name_act[i] + ') : ', funcs.MSE(z_tilde, z_test))

        mse_train, mse_test = train_network_reg(network, bsize, n_epochs, eta,
                                                X_train, X_test, z_train, z_test)


        # Runs trained neural network with testing data
        z_tilde_train = network.feed_forward(X_train)
        z_tilde_test = network.feed_forward(X_test)

        print(f"MSE after training train ({name_act[i]}): {funcs.MSE(z_tilde_train, z_train)}")
        print(f"MSE after training test ({name_act[i]}): {funcs.MSE(z_tilde_test, z_test)}")

        plot_results(mse_test, name_act[i],
                    'Epochs', 'MSE', 'Testing Activations functions',
                    ylim=[0,1], filename=filename)
    plt.show()


    return mse_train, mse_test
