import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.models import Sequential      #This allows appending layers to existing models
from tensorflow.keras import optimizers, regularizers          #This allows using whichever optimiser we want (sgd,adam,RMSprop)
from tensorflow.keras.utils import to_categorical   #This allows using categorical cross entropy as the cost function
from tensorflow.keras import datasets, layers, models
import numpy as np

from FruitReads import read_fruit

tf.autograph.set_verbosity(1)


'''
This file contains the class Kreas that implements neural network form tensorflow/kreas.

'''

#Create a CNN class with function from lecture notes week 42
class CNN_Keras:
    def __init__(self, input_shape, receptive_field, n_filters, n_neurons_connected, n_categories, eta, lmbd, act):
        """
        Set up neural network using keras.

        Agruments:
            input_shape (int/tuple?): The shape of the input data  
            receptive_field (int/list/tuple): size of receptive field // kernel size
            n_filters (int): number of filters/fields // number of output filters in the convolution(?)
            n_neurons_connected (int): number of neurons in dense layer 
            n_categories (int): Number of categories
            eta (float): learning rate 
            lmbd (float): lambda , regularization?
        """

        self.input_shape = input_shape
        self.receptive_field = receptive_field
        self.n_filters = n_filters
        self.n_neurons_connected = n_neurons_connected
        self.n_categories = n_categories 
        self.eta = eta
        self.lmbd = lmbd

        self.act = act


    #Modydied from lecture notes (week 42)
    def CNN_model(self):

        '''
        Create a convolutional neural network
        '''

        model=Sequential()

        model.add(layers.Conv2D(self.n_filters, (self.receptive_field, self.receptive_field), 
                                input_shape=self.input_shape, padding='same',
                                activation=self.act, kernel_regularizer=regularizers.l2(self.lmbd)))
        

        # Pooling layer 
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        # Flatten before the fully connected dense layers 
        model.add(layers.Flatten())

        # Apply elementwise activation function 
        model.add(layers.Dense(self.n_neurons_connected, activation=self.act, kernel_regularizer=regularizers.l2(self.lmbd)))
       
        # outpulayer/fully-connected layer 
        model.add(layers.Dense(self.n_categories, activation='softmax', kernel_regularizer=regularizers.l2(self.lmbd)))
        
        sgd = optimizers.SGD(learning_rate=self.eta)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
        
        return model
