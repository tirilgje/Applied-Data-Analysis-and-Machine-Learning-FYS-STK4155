import numpy as np



'''
This file contains implementations of the cost functions used. 
Each cost function is represented as a class, 
the __call__ function returns the input with the function added, and the deriv function returns the derivative
The cost functions implemented are: MSE and CrossEntropy
'''

class MSE:
    """Calculates the mean squared error cost function and its derivative."""
    def __call__(self, z_tilde, z):
        """Calculates the mean squared error.

        Arguments:
            z_tilde (matrix): predicted data
            z (matrix): true data
        Returns:
            MSE (float): mean squared error of predicted and true data
        """

        n = np.size(z)
        return np.sum((z_tilde-z)**2)/n


    def deriv(self, z_tilde, z):
        """Computes the derivative of mean squared error cost function.
        Arguments:
            z_tilde (matrix): predicted data
            z (matrix): true data
        Return value:
            deriv (float?): the derivative of the mean squared error
        """
        deriv = 2*(z_tilde - z)/z.shape[0]
        return deriv

class CrossEntropy:
    '''
    Calculates the categorical cross entropy cost function and its derivative.
    '''
    def __call__(self, z_tilde, z):
        '''
        Calculates the categorical cross entropy.
        Arguments:
            z_tilde (matrix): predicted data
            z (matrix): true data
        Return value:
            cross_entropy (matrix): categorical cross entropy of predicted and true data
        '''
        cross_entropy = -np.log(np.prod(np.pow(z_tilde,z)))
        return cross_entropy

    def deriv(self, z_tilde, z):
        '''
        Computes the derivative of categorical cross entropy cost function.
        Arguments:
            z_tilde (matrix): predicted data
            z (matrix): true data
        Return value:
            deriv (matrix): the derivative of the categorical cross entropy cost function
                     when using softmax as the output activation function
        '''
        deriv = z_tilde - z
        return deriv

