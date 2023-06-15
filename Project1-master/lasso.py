import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns
import pandas as pd
from sklearn import linear_model


import bootstrap as bt
import cross_validation as cv
import funcs
import linear_regression


#Lasso analysis


def lasso_without_resampling(x,y,z,n):
    """
    Does lasso analysis without any resamplings techniques
    
    """
    #different order 
    for i in range(3, 10, 2):
        
        data = funcs.prep_linear_regression(x, y, z, i)

        X_train_scaled = data["X_train_scaled"]
        X_test_scaled = data["X_test_scaled"]
        z_train_scaled = data["z_train_scaled"]
        z_test_scaled = data["z_test_scaled"]
        
        # Decide which values of lambda to use
        nlambdas = 10
        lambdas = np.logspace(-5, 1, nlambdas)
        MSE_test = np.zeros(nlambdas)

        order = i

        lambdas = np.logspace(-10, 1, nlambdas)

        for j in range(nlambdas):
            
            lmb = lambdas[j]
        
            #lasso using Scikit-Learn
            RegLasso = linear_model.Lasso(lmb, fit_intercept = False)
            RegLasso.fit(X_train_scaled, z_train_scaled)

            # and then make the prediction
            z_tilde_train = RegLasso.predict(X_train_scaled)
            z_tilde_test = RegLasso.predict(X_test_scaled)

            MSE_test[j] = funcs.MSE(z_test_scaled, z_tilde_test)
            
        # Now plot the results
        #plt.figure()
        plt.plot(np.log10(lambdas), MSE_test, label = 'MSE Lasso Test ' + str(i))

    plt.xlabel('log10(lambda)')
    plt.ylabel('Prediction Error')
    plt.yscale('log')
    plt.title('Lasso without resampling, n = ' + str(n**2))
    plt.legend()
    plt.show()
    
    
    
def lasso_crossvalidation(x,y,z,n):
    
    
    """
    Cross validation analysis for Lasso 
    
    """
    method = 'Lasso' 
    n_folds = 8
    
    #different order 
    for order in range(3, 10, 2):
        # Decide which values of lambda to use
        nlambdas = 10
        lambdas = np.logspace(-5, 1, nlambdas)
        MSE_test = np.zeros(nlambdas)

        for k in range(nlambdas):
            lmd = lambdas[k]
            data  = cv.cross_validation_1run(x, y, z, n, order, n_folds, seed, method, nlambda=lmd)
                        
            MSE_test[k] = data["MSE_test"]

        # Now plot the result
        plt.plot(np.log10(lambdas), MSE_test, label = 'MSE Lasso Test, degree ' + str(order))
        plt.xlabel('log10(lambda)')
        plt.ylabel('Prediction Error')
        plt.title("Cross validation Lasso, n = " + str(n**2))
        plt.legend()
    plt.show()
    
    
    
def lasso_bootstrap(x,y,z,n):
    
    """
    Bootstrap analysis for Lasso
    """
    method = 'Lasso'
    nboot = 50
    order = 8
    

    # Decide which values of lambda to use
    nlambdas = 10
    lambdas = np.logspace(-5, 1, nlambdas)
    MSE_test = np.zeros(nlambdas)
    bias_bootstrap = np.zeros(nlambdas)
    variance_bootstrap = np.zeros(nlambdas)
    MSE_test = np.zeros(nlambdas)

    for k in range(nlambdas):
        lmd = lambdas[k]
        data  = bt.bootstrap_1run(x, y, z, order, nboot, method, nlambda=lmd)

        bias_bootstrap[k] = data["bias_bootstrap"]
        variance_bootstrap[k] = data['variance_bootstrap']
        MSE_test[k] = data['MSE_bootstrap_test']
            
            
    # Now plot the result
    plt.plot(np.log10(lambdas), bias_bootstrap, label = 'bias')

    plt.plot(np.log10(lambdas), variance_bootstrap, label = 'variance')
    plt.plot(np.log10(lambdas), bias_bootstrap+variance_bootstrap, label = 'bias+variance')
    plt.plot(np.log10(lambdas), MSE_test, '--', label = 'MSE')

    plt.xlabel('log10(lambda)')
    plt.ylabel('Prediction Error')
    plt.title("Lasso, Bias-variance trade-off, n = " + str(n**2))
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    
    
def test(x,y,z,n): 
    
    
    """
    Test lasso with lasso scaler 
    
    """
    #different order 
    for i in range(3, 10, 2):
    
        X = funcs.create_X(x, y, n)
        X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)


        # Decide which values of lambda to use
        nlambdas = 10
        lambdas = np.logspace(-5, 1, nlambdas)
        MSE_test = np.zeros(nlambdas)

        order = i

        lambdas = np.logspace(-10, 1, nlambdas)

        for j in range(nlambdas):

            lmb = lambdas[j]

            #lasso using Scikit-Learn
            RegLasso = linear_model.Lasso(lmb, normalize = True)
            RegLasso.fit(X_train, z_train)

            # and then make the prediction
            z_tilde_train = RegLasso.predict(X_train)
            z_tilde_test = RegLasso.predict(X_test)

            MSE_test[j] = funcs.MSE(z_test, z_tilde_test)

        # Now plot the results
        #plt.figure()
        plt.plot(np.log10(lambdas), MSE_test, label = 'MSE Lasso Test ' + str(i))

    plt.xlabel('log10(lambda)')
    plt.ylabel('Prediction Error')
    plt.yscale('log')
    plt.title('Lasso normalize with sklearn, n = ' + str(n**2))
    plt.legend()
    plt.show()
    


             

#choose data to analyse         
franke = False  
terrain = True

n = 20       #number of datapoints = n**2 
seed = 111   #using this seed in all analysis

#the data 
x, y, z = funcs.make_data(n, seed=seed)

n_terrain = 20
x_terrain, y_terrain, z_terrain = funcs.make_data(n, seed, filename = 'SRTM_data_Norway_1.tif')



if franke: 
    lasso_without_resampling(x,y,z,n)
    lasso_crossvalidation(x,y,z,n)
    lasso_bootstrap(x,y,z,n)
    test(x,y,z,n)
    
if terrain:
    
    lasso_without_resampling(x_terrain,y_terrain,z_terrain,n_terrain)
    lasso_crossvalidation(x_terrain,y_terrain,z_terrain,n_terrain)
    lasso_bootstrap(x_terrain,y_terrain,z_terrain,n_terrain)