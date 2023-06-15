import numpy as np

'''
This file contains implementations of the Activation functions used.
Each aktivationfunction is represented as a class,
the __call__ function returns the input with the function added, and the deriv function returns the derivative
The aktivation functions implemented are: Sigmoid, ReLU, Leaky ReLU and Identity.
'''

class Activation_ReLU:

    # Feed forward
    def __call__(self, inputs):

        '''
        ReLU
        Set all negative values to 0
        Arguments:
            inputs (matrix): the input data
        Returns:
            (matrix): same shape as inputs with the function added
        '''
        return np.maximum(0, inputs)

    # Backpropagation function
    def deriv(self, inputs):
        '''
        Derivative of ReLU
        Sets all negative values to 0 and positive values to 1.
        Agruments:
            inputs (matrix): the input data
        Returns:
            (matrix): same shape as inputs with the function added
        '''
        return np.where(self(inputs) > 0, 1, self(inputs))

class Activation_LeakyReLU:

    # Feed forward
    def __call__(self, inputs):
        '''
        Leaky ReLU
        Sets all negative values z to z*0.01
        Arguments:
            inputs (matrix): The input data
        Returns:
            (matrix): same shape as inputs with the function added
        '''
        return np.where(inputs > 0, inputs, inputs * 0.01)

    # Backpropagation function
    def deriv(self, inputs):
        '''
        Derivative of Leaky ReLU
        Sets all positive values to 1 and negative values to 0.01
        Arguments:
            inputs (matrix): the input data
        Returns:
            (matrix): same shape as inputs with the function added
        '''
        return np.where(inputs > 0, 1, 0.01)




class Activation_Sigmoid:

    # Feed forward
    def __call__(self, inputs):
        '''
        Sigmoid
        Calcultes the sigmoid function on the input
        Arguments:
            inputs (matrix): The input data
        Returns:
            (matrix): same shape as inputs with the function added
        '''
        import sys
        def less_than_0(inputs):
            a = np.exp(inputs.astype(float))
            #if np.amax(a) > 100000000:
            #    print("HER HØYE TALL SIGMOID KRISE",np.amax(a))
            #    sys.exit()
            a = a/(1+a)
            return a

        def greater_than_0(inputs):
            return 1/(1 + np.exp(-inputs.astype(float)))

        #print(np.mean(inputs))

        return(np.where(inputs > 0, greater_than_0(inputs), less_than_0(inputs)))


    # Backpropagation function
    def deriv(self, inputs):
        '''
        Derivative of sigmoid
        Arguments:
            inputs (matrix): The input data
        Returns:
            (matrix): same shape as inputs with the function added
        '''
        return self(inputs) * (1-self(inputs))


class Activation_Identity:

    def __call__(self, inputs):
        '''
        Identity function returns the input as it is.
        Argumensts:
            inputs (matrix): The input data
        Returns:
            (matrix): same shape as inputs with the function added
        '''
        return inputs

    def deriv(self, inputs):
        '''
        Derivative if Identity, returns all values as 1.
        Argumensts:
            inputs (matrix): The input data
        Returns:
            (matrix): same shape as inputs with the function added
        '''
        return np.ones(inputs.shape)

class Activation_Softmax:

    def __call__(self, z):
        # E
        # max_z = np.max(z, axis=1).reshape(-1, 1)
        # softmax = np.exp(z - max_z)/np.sum(np.exp(z - max_z), axis=1)[:, None]

        # Youtubemannen
        exp_values = np.exp(z - np.max(z, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        # self.output = probabilities
        return probabilities

    def deriv(self, z):
        # E - gadd ikke derivere og dobbeltsjekke selv tbh
        deriv = self.__call__(z) - self.__call__(z)**2
        return deriv
