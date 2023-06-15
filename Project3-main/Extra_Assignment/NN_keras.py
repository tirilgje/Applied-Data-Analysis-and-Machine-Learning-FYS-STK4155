import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras.layers import Dense           #This allows defining the characteristics of a particular layer
from tensorflow.keras import optimizers             #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras import regularizers           #This allows using whichever regularizer we want (l1,l2,l1_l2)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
import numpy as np

tf.autograph.set_verbosity(1)


'''
This file contains the class Kreas that implements neural network form tensorflow/kreas.
'''

#Create a class with function from lecture notes
class NN_Keras:
    def __init__(self, n_in_each_hidden, n_in, n_out, eta, lmd, loss, metrics, act_hidden,
                 act_out):
        """
        Set up neural network using keras.

        Agruments:
            n_in_each_hidden (list): list with number of neurons in each hidden layer
            n_in (int): number of inputs to the network
            n_out (int): number of outputs from the network
            eta (float): learning rate
            lmd (float): regulariztion parameter
            loss (class): loss function
            metrics (matrix): metrics (??)
            act_hidden (string): name of the activation function in the  hidden layers
            act_out (string): name of the activation function in the output layer
        """
        self.n_hidden = n_in_each_hidden
        self.n_out = n_out
        self.n_in = n_in
        self.eta = eta
        self.lmd = lmd
        self.loss = loss
        self.metrics = metrics
        self.act_hidden = act_hidden
        self.act_out = act_out


    #Modydied from lecture notes (week 41)
    def NN_model(self):

        '''
        Create a neural network
        '''

        model=Sequential()

        model.add(Dense(self.n_hidden[0], activation=self.act_hidden,
                        kernel_regularizer=regularizers.l2(self.lmd),input_dim=self.n_in))


        for i in range(1, len(self.n_hidden)):
            model.add(Dense(self.n_hidden[i],activation=self.act_hidden,kernel_regularizer=regularizers.l2(self.lmd)))


        model.add(Dense(self.n_out, activation=self.act_out))
        sgd=optimizers.SGD(learning_rate=self.eta)
        model.compile(loss=self.loss,optimizer=sgd,metrics=self.metrics)

        return model
