import numpy as np
import funcs

'''
This file contains the implementation of SGD. 
SGD is implemented as a class with the following functions:
    gradient_cost: calculates the gradient 
    learning_schedule: calculates a learning rate when dynamic is choosen
    SGD: The SGD algorithm 

'''


class StocasticGradientDescent:
    def __init__(self, X_train, X_test, z_train,z_test):
        # Setter opp X og z for å brukes pent i klassen
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

        self.n = X_train.shape[0]


    def gradient_cost(self, Xb, zb, theta, bsize, lmb = 0):
        """
        lmb = 0 gir OLS, men gjør klar for å bruke Ridge senere
            Note: gradient mangler fortsatt et lambda-ledd, men det lå ikke
            i den spesifikke funksjonen jeg brukte. (se nedenfor)

        Hentet fra forelesning 39 under tittelen (faktisk implementert teori)
            "The derivative of the cost/loss function"
        Se også under tittelen  (Eksempelkode som implementerer teori)
            "Gradient Descent Example"
        For mulig ridge implementation se her: (Legger på lambda for ridge)
            "Gradient descent and Ridge"


        Calculates the gradient step in one interation in one epoch
        from "program for SGD"-code week 40
        """
        gradient = 2/bsize * Xb.T @ (Xb @ theta - zb) + 2*lmb * theta

        return gradient


    def learning_schedule(self, t, t0, t1):
        """
        Fra eksempelkoden i sgd greia i funcs
        funksjon fra forelesningsnotater, fant en implementering under
            "Slightly different approach"
        """
        return t0/(t+t1)


    def SGD(self, n_epochs, bsize, theta=None, t0=5, t1=50, gamma=0.01, _eta=None, lmb = 0):
        """
        Arguments:
            X: design matrix, y: predicted values
            n_epochs: number of epochs
            bsize: size of each minibatch
            t0, t1: learning stuff dynamic learning rate
            gamma: momentum stuff
            eta: learning rate

            from "program for SGD"-code week 40
        """
        X_train = self.X_train
        z_train = self.z_train
        n = self.n
        v=0

        np.random.seed(123)


        minibatches = int(n/bsize) #number of minibatches

        # If theta is not given, use a random to start with 
        if theta == None:
            theta = np.random.randn(X_train.shape[1], 1)

        # For saving mse values 
        self.mse = np.zeros(n_epochs-1)

        index = np.arange(n)
        for epoch in range(1,n_epochs):
            np.random.shuffle(index)
            Xbatches = np.array_split(X_train[index], minibatches)
            zbatches = np.array_split(z_train[index], minibatches)

            for i in range(minibatches):

                r = np.random.randint(0, minibatches)
                gradient = self.gradient_cost(Xbatches[r], zbatches[r], theta, bsize, lmb)
                if _eta == None:
                    eta = self.learning_schedule(epoch*n, t0, t1)
                else:
                    eta = _eta
                v = gamma * v + eta*gradient    
                theta = theta - v               

            z_tilde = self.X_test @ theta
            self.mse[epoch-1] = funcs.MSE(self.z_test, z_tilde)

        return theta
