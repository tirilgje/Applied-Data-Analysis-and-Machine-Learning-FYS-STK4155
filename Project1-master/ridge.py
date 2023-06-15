import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import seaborn as sns
import pandas as pd


import bootstrap as bt
import cross_validation as cv
import funcs
import linear_regression

#Does most of the analysis for ridge regression 

def ridge_without_resampling(x,y,z,n):
    
    """
    
    Ridge regression without any resampling techniques
    
    """
    method = "calc_Ridge" 
    
    #different order 
    for i in range(3, 10, 2):
        
        # Decide which values of lambda to use
        nlambdas = 10
        lambdas = np.logspace(-10, 1, nlambdas)
        MSE_test = np.zeros(nlambdas)


        order = i

        for j in range(nlambdas):
            lmd = lambdas[j]
            data, r2mse = linear_regression.linear_regression(x, y, z, order=order, lmd=lmd, method=method)
            MSE_test[j] = r2mse["mse_test"]

        # Now plot the results
        plt.plot(np.log10(lambdas), MSE_test, label = 'MSE Ridge Test, degree ' + str(i))
        plt.xlabel('log10(lambda)')
        plt.ylabel('Prediction Error')
        plt.title("MSE depending on lambda and complexity for Ridge, n = " + str(n**2))
        plt.legend()
    plt.show()
    
    
    
def ridge_crossvalidation(x,y,z,n):
    
    """
    Cross validation analysis for ridge regression
    """
    method = funcs.calc_Ridge 
    n_folds = 8
    
    #different order 
    for order in range(3, 10, 2):
        # Decide which values of lambda to use
        nlambdas = 10
        lambdas = np.logspace(-10, 1, nlambdas)
        MSE_test = np.zeros(nlambdas)

        for k in range(nlambdas):
            lmd = lambdas[k]
            data  = cv.cross_validation_1run(x, y, z, n, order, n_folds, seed, method, nlambda=lmd)
            
            MSE_test[k] = data["MSE_test"]

        # Now plot the result
        plt.plot(np.log10(lambdas), MSE_test, label = 'MSE Ridge Test, degree ' + str(order))
        plt.xlabel('log10(lambda)')
        plt.ylabel('Prediction Error')
        plt.title("Cross validation for Ridge, n = " + str(n**2))
        plt.legend()
    plt.show()
    
    

    
    
def ridge_bootstrap(x,y,z,n):
    
    """
    Bootstrap analysis for redge regression 
    """
    method = funcs.calc_Ridge 
    nboot = 50
    order = 7
    

    # Decide which values of lambda to use
    nlambdas = 10
    lambdas = np.logspace(-10, 1, nlambdas)
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
    plt.plot(np.log10(lambdas), bias_bootstrap, label = 'bias^2')

    plt.plot(np.log10(lambdas), variance_bootstrap, label = 'variance')
    plt.plot(np.log10(lambdas), bias_bootstrap+variance_bootstrap, label = 'bias^2+variance')
    plt.plot(np.log10(lambdas), MSE_test, '--', label = 'MSE')

    plt.xlabel('log10(lambda)')
    plt.ylabel('Prediction Error')
    plt.title("Bias-variance trade-off for Ridge, n = " + str(n**2))
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    
    
    
def ridge_cross_nfold(x,y,z,n):
    
    """
    Cross validation analysis for ridge regression 
    
    """
    
    seed = 111
    degree = 20


    method = funcs.calc_Ridge
    
    for i in range(5,11,1):
        if (n**2)%i == 0:
            n_folds = i   # number of folds
            data_cv = cv.cross_validation(x, y, z, n, degree, n_folds, seed, method)

            polydegree = data_cv["polydegree"]
            MSE_cv = data_cv["MSE_test"]

            plt.plot(polydegree, MSE_cv, label='k fold = ' + str(i))
        
    plt.xlabel("Model complexity (polynomial degree)")
    plt.ylabel("Prediction error")
    plt.title("MSE for Ridge crossvalidation")
    plt.yscale("log")
    plt.legend()
    plt.show()
             
#choose data to look at
terrain = False 
franke = True 

n = 20       #number of datapoints = n**2 
seed = 111   #using this seed in all analysis

#the data 
x, y, z = funcs.make_data(n, seed=seed)

n_terrain = 20
x_terrain, y_terrain, z_terrain = funcs.make_data(n, seed = seed, filename = 'SRTM_data_Norway_1.tif')


if terrain: 
    
    ridge_without_resampling(x_terrain,y_terrain,z_terrain,n_terrain)
    ridge_crossvalidation(x_terrain,y_terrain,z_terrain,n_terrain)
    ridge_bootstrap(x_terrain,y_terrain,z_terrain,n_terrain)
    
if franke: 
    
    ridge_without_resampling(x,y,z,n)
    ridge_crossvalidation(x,y,z,n)
    ridge_bootstrap(x,y,z,n)
    ridge_cross_nfold(x,y,z,n)
             