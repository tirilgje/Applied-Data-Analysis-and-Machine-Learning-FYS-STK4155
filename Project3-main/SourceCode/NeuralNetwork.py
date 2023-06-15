import numpy as np
import activation as act

# sets a global random seed for the script 
np.random.seed(1234)

'''
This file contains the implementation of the neural network. 
It contains two classes: Layer and NeuralNetwork

The neural network is implemented as a class.
The class contains the following functions: 
    create_layers: building up the nettwork with layers 
    feed_forward: takes the inputs throught the network and produsing the output. 
    backprop: the backpropagation including updating weights and biases
    logistic: implementation of logistic regression

Each layer in the class is represented as an instance of the class Layer containg information about the layer. 

'''


class Layer:
    def __init__(self, n_in, n_out, activation):
        '''
        One layer of the neural network

        Arguments:
            n_in (int): number of input nodes of the layer
            n_out (int): number of output nodes of the layer
            activation (class): the actication function to use on the output
        '''

        self.n_in = n_in
        self.n_out = n_out
        self.act = activation


        # n x m matrix of random floats for the weights
        # sampled from a normal distribution with mean 0 and variance 1.
        self.weight = np.random.randn(self.n_in, self.n_out)

        #trying to optimize the inisialization of the weights
        if isinstance(activation, act.Activation_Sigmoid):
            self.weight = self.weight * np.sqrt(1.0/(n_in))
        
        elif isinstance(activation, act.Activation_ReLU) or isinstance(activation, act.Activation_LeakyReLU):
            self.weight = self.weight * np.sqrt(2.0/n_in)

        # 1 x m matrix of random bias to be added
        self.bias = 0.001 * np.ones((1, n_out))

    def __call__(self, X):

        '''
        Runs throught one layer and retuns the output.
        Does the feed forward prosess for given layer.

        Arguments:
            X (matrix): the input data

        Returns:
            out (matrix): the output with the activationfunction applied
        '''

        # Calculate the z values for the layer by mulitiplying input with the weights and then add the bias
        # print('call layer forward:', X.shape, self.weight.shape)
        
        self.z_tilde = X @ self.weight + self.bias

        # adding the activation function
        self.out = self.act(self.z_tilde)
        #print(self.out)
        self.out_deriv = self.act.deriv(self.z_tilde)

        return self.out


class NeuralNetwork:
    def __init__(
            self,
            n_in,
            n_out,
            activation,
            activation_out,
            costfunc,
            n_in_each_hidden = None):


        '''
        Setup for the neural network

        Arguments:
            n_in (int): Number of nodes in input layer
            n_out (int): Number of nodes in output layer
            n_in_each_hidden (list) [optional, default = None]:
                        List containing number of nodes in each of the hidden layers.
                        If None there is no hidden layers
            activation (class): the activation function
        '''

        self.n_in = n_in
        self.n_out = n_out
        self.n_in_each_hidden = n_in_each_hidden
        self.act = activation
        self.act_out = activation_out
        self.costfunc = costfunc


    def create_layers(self):
        '''
        Create all the layers of the neural newwork

        Arguments:

        '''

        n_in = self.n_in
        n_out = self.n_out
        self.layers = []

        if self.n_in_each_hidden:
            #make layers including the hidden ones

            # add first layer
            first_layer = Layer(self.n_in,
                                self.n_in_each_hidden[0],
                                self.act)

            self.layers.append(first_layer)

            #add all the hidden layers
            for i in range(len(self.n_in_each_hidden) - 1):
                hidden_layer = Layer(self.n_in_each_hidden[i],
                                     self.n_in_each_hidden[i+1],
                                     self.act)

                self.layers.append(hidden_layer)

            #add output layer
            output_layer = Layer(self.n_in_each_hidden[-1],
                                 self.n_out,
                                 self.act_out)

            self.layers.append(output_layer)


        else:
            #direclty from input to output layer (used for implementing logistic regression)
            layer = Layer(n_in, n_out, self.act_out)
            self.layers.append(layer)

        return self.layers


    def feed_forward(self, X):
        '''
        Feed forward

        Arguments:
            X (matrix): The input data (x,y)

        Returns:
            X (matrix): The output data after a forward run (z)
        '''
        for layer in self.layers:
            X = layer(X)

        return X


    def backprop(self, X, z, eta = 0.001, lmd = 0):
        """
        Backpropagation function:
        Step 1: Implement the feed forward algorithm.
        Step 2: Calculates the output error from the feed forward algorithm.
        Step 3: Reversed loop through the layers to update the weights and biases.

        Arguments:
            X (matrix): Input data
            z (matrix): True output data
            eta (float): learning rate
            lmbda (float): regularization parameter
        """
        self.feed_forward(X)
        layers = self.layers

        # Derivatve of the costfunction
        dCda = self.costfunc.deriv(layers[-1].out, z)
        # Calculate the output error of the output layer, dL
        dL = dCda * layers[-1].out_deriv
        # Note: out and out_deriv is denoted as a and a' in lecture notes (week 41)

        #Update weights between the outputlayer and the last hidden layer
        layers[-1].weight = layers[-1].weight - eta*(layers[-2].out.T @ dL)
        layers[-1].bias = layers[-1].bias - eta*dL[0, :]

        # Between the hidden layers 
        for l in reversed(range(1, len(layers) - 1)):
            # Calculates back propagate error from last hidden layer to second first hidden layer
            dL = (dL @ layers[l + 1].weight.T) * layers[l].out_deriv

            # Updates the weights
            layers[l].weight =  layers[l].weight - eta*(layers[l - 1].out.T @ dL)- 2*eta*lmd*layers[l].weight 

            # Updates the biases
            layers[l].bias = layers[l].bias - eta*dL[0, :]

        # calculate dL for first hidden layer
        dL = (dL @ layers[1].weight.T) * layers[0].out_deriv
        # Updates the weights
        layers[0].weight = layers[0].weight - eta*(X.T @ dL) - 2*eta*lmd*layers[0].weight

        #print('weight', layers[0].weight)
        # Updates the biases
        layers[0].bias = layers[0].bias - eta*dL[0, :]


    def logistic(self, X, z, eta = 0.01, lmbda = 0.1):
        """
        Using the code from neural network to do the logistic regression 
        Here we do not use any hidden layers

        Arguments:
            X (matrix): Input data
            z (matrix): True output data
            eta (float): learning rate
            lmbda (float): regularization parameter
        """
        self.feed_forward(X)
        layers = self.layers

        # Derivatve of the costfunction
        dCda = self.costfunc.deriv(layers[-1].out, z)
        # Calculate the output error of the output layer, dL
        dL = dCda * layers[0].out_deriv
        # Note: out and out_deriv is denoted as a and a' in lecture notes (week 41)

        # Updates the weights
        layers[0].weight = layers[0].weight - eta*(X.T @ dL) - 2*eta*lmbda*layers[0].weight
        # Updates the biases
        layers[0].bias = layers[0].bias - eta*dL[0, :]
