import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

import funcs
import linear_regression
import bootstrap as bt
import cross_validation as cv 

#bruker seed 111 for analyse i rapport


#Does most of the ols analysis

def ols(x,y,z,n):
    
    
    """
    OLS without resampling 
    
    """
    #Testing ols for different polynomial degree
    degree = 25

    MSE_test = np.zeros(degree)
    MSE_train = np.zeros(degree)
    R2_test = np.zeros(degree)
    R2_train = np.zeros(degree)
    polydegree = np.zeros(degree)

    #Plot MSE for OLS (without resampling) for different complexity
    for i in range(degree):
        polydegree[i] = i

        data, r2mse = linear_regression.linear_regression(x, y, z, order=i, method="calc_OLS")    

        MSE_train[i] = r2mse["mse_train"]
        MSE_test[i] = r2mse["mse_test"]

    title = "Comparison of MSE for test and train data, n = " + str(n**2)
    funcs.plot_ols(polydegree, MSE_test, MSE_train, title)


    #Choose one order and look at the scores and CI
    data, r2mse = linear_regression.linear_regression(x, y, z, order=5, method="calc_OLS")    

    #Looking at R2 and MSE 
    R2_train = r2mse["r2_train"]
    R2_test = r2mse["r2_test"]
    MSE_train = r2mse["mse_train"]
    MSE_test = r2mse["mse_test"]

    print("MSE and R2 for a fifth order polynomial fit with OLS:")
    print("Train: \nMSE: ", MSE_train, "\nR2:  " ,R2_train)
    print("\nTest: \nMSE: ", MSE_test, "\nR2:  " ,R2_test)


    #For confidenceintervals 
    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    z_train_scaled = data["z_train_scaled"]
    z_test_scaled = data["z_test_scaled"]


    #Plot this?
    confidenceinterval = funcs.confidence_interval(X_train_scaled, z_train_scaled)
    





def cross_vs_boot(x,y,z,n):
    
    """
    Bootstrap analysis for OLS 
    """
    
    
    seed = 111
    degree = 20
    n_folds = 5   # number of folds
    nboot = 50


    method = funcs.calc_OLS

    data_cv = cv.cross_validation(x, y, z, n, degree, n_folds, seed, method)
    data_bt = bt.bootstrap_analysis(x, y, z, degree, nboot, method)


    polydegree = data_cv["polydegree"]
    MSE_cv = data_cv["MSE_test"]
    MSE_bt = data_bt["MSE_bootstrap_test"]    
    
    
    plt.plot(polydegree, MSE_cv, label='MSE bootstrap')
    plt.plot(polydegree, MSE_bt, label='MSE cross-validation')
    plt.xlabel("Model complexity (polynomial degree)")
    plt.ylabel("Prediction error")
    plt.title("Comparison of MSE using bootstrap and cross-validation on OLS")
    plt.yscale("log")
    plt.legend()
    plt.show()
    




def cross_nfold(x,y,z,n):
    
    """
    Cross validation analysis for ols 
    """
    
    seed = 111
    degree = 20

    method = funcs.calc_OLS
    
    for i in range(5,11,1):
        if (n**2)%i == 0:
            n_folds = i   # number of folds
            data_cv = cv.cross_validation(x, y, z, n, degree, n_folds, seed, method)

            polydegree = data_cv["polydegree"]
            MSE_cv = data_cv["MSE_test"]

            plt.plot(polydegree, MSE_cv, label='k fold = ' + str(i))
        
    plt.xlabel("Model complexity (polynomial degree)")
    plt.ylabel("Prediction error")
    plt.title("MSE for OLS crossvalidation")
    plt.yscale("log")
    plt.legend()
    plt.show()
    

        
#choose data to look at 
franke = False 
terrain = True 
    
#The data 
n = 20
x, y, z = funcs.make_data(n, seed=111)

n_terrain = 20
x_terrain, y_terrain, z_terrain = funcs.make_data(n_terrain, seed = 111, filename = 'SRTM_data_Norway_1.tif')

if franke:
    ols(x,y,z,n)
    cross_vs_boot(x,y,z,n)
    cross_nfold(x,y,z,n)
    
    
if terrain: 
    ols(x_terrain,y_terrain,z_terrain, n_terrain)
    cross_vs_boot(x_terrain, y_terrain, z_terrain, n_terrain)
    cross_nfold(x_terrain, y_terrain, z_terrain, n_terrain)
