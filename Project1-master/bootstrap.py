import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import Ridge, Lasso
from sklearn import linear_model

import funcs
import linear_regression


def bootstrap_analysis(x,y,z, degree, nboot, method, nlambda=0):

    """
    Does bootstrap analysis on given data with given arguments for given method
    Returns arrays with calulated MSE, bias and variance.

    Input:
        x,y,z (arrays): the data
        degree(int): polynomdegree
        nboot(int): number of resample(?)
        method: the method we shuld use to calculate beta (ols, ridge or lasso)

    Output: (arrays of lenth degree)
        polydegree: ints from 0 to degree
        MSE_bootstrap_test: MSE values for test-data (for degree 0 to degree)
        MSE_bootstrap_train: MSE values for train-data (for degree 0 to degree)
        bias_bootstrap: the calculated bias (for degree 0 to degree)
        variance_bootstrap: the calculated variansce (for degree 0 to degree)
    """

    MSE_bootstrap_test = np.zeros(degree)
    MSE_bootstrap_train = np.zeros(degree)
    bias_bootstrap = np.zeros(degree)
    variance_bootstrap = np.zeros(degree)
    polydegree = np.zeros(degree)

    for i in range(degree):
        polydegree[i] = i+1

        data = funcs.prep_linear_regression(x, y, z, order=i+1)

        X_train_scaled = data["X_train_scaled"]
        X_test_scaled = data["X_test_scaled"]
        z_train_scaled = data["z_train_scaled"]
        z_test_scaled = data["z_test_scaled"]


        z_tilde_test = np.zeros((np.shape(z_test_scaled)[0], nboot))
        z_tilde_train = np.zeros((np.shape(z_train_scaled)[0], nboot))


        index = np.arange(0, X_train_scaled.shape[0], 1)

        for j in range(nboot):
            id = resample(index) #sjekk at resample er med replacement

            z_train_scaled_ = z_train_scaled[id]
            X_train_scaled_ = X_train_scaled[id, :]

            if method == funcs.calc_OLS:
                beta = method(X_train_scaled_, z_train_scaled_)
                z_tilde_test[:, j] = np.ravel(X_test_scaled @ beta)
                z_tilde_train[:, j] = np.ravel(X_train_scaled @ beta)
            elif method == funcs.calc_Ridge:
                beta = method(X_train_scaled_, z_train_scaled_, nlambda)
                z_tilde_test[:, j] = np.ravel(X_test_scaled @ beta)
                z_tilde_train[:, j] = np.ravel(X_train_scaled @ beta)
            elif method == "Lasso":
                clf = linear_model.Lasso(alpha = nlambda)
                clf.fit(X_train_scaled_, z_train_scaled_)
                z_tilde_test[:, j] = clf.predict(X_test_scaled)
                z_tilde_train[:, j] = clf.predict(X_train_scaled)



        MSE_bootstrap_test[i] = np.mean(np.mean((z_test_scaled - z_tilde_test)**2,
                                                axis=1,
                                                keepdims=True))

        MSE_bootstrap_train[i] = np.mean(np.mean((z_train_scaled - z_tilde_train)**2,
                                                 axis=1,
                                                 keepdims=True))

        bias_bootstrap[i] = np.mean((z_test_scaled - np.mean(z_tilde_test,
                                                             axis=1,
                                                             keepdims=True))**2)

        variance_bootstrap[i] = np.mean(np.var(z_tilde_test,
                                               axis=1,
                                               keepdims=True))

    data = {"polydegree":polydegree,
            "MSE_bootstrap_test":MSE_bootstrap_test,
            "MSE_bootstrap_train":MSE_bootstrap_train,
            "bias_bootstrap":bias_bootstrap,
            "variance_bootstrap":variance_bootstrap}


    return data



def bootstrap_1run(x,y,z, polydegree, nboot, method, nlambda=0):

    """
    the same algoritm as bootstrap analysis, but just one run, not for several complexities.

    """  

    data = funcs.prep_linear_regression(x, y, z, order=polydegree)

    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    z_train_scaled = data["z_train_scaled"]
    z_test_scaled = data["z_test_scaled"]


    z_tilde_test = np.zeros((np.shape(z_test_scaled)[0], nboot))
    z_tilde_train = np.zeros((np.shape(z_train_scaled)[0], nboot))


    index = np.arange(0, X_train_scaled.shape[0], 1)

    for j in range(nboot):
        id = resample(index) #sjekk at resample er med replacement

        z_train_scaled_ = z_train_scaled[id]
        X_train_scaled_ = X_train_scaled[id, :]

        if method == funcs.calc_OLS:
            beta = method(X_train_scaled_, z_train_scaled_)
            z_tilde_test[:, j] = np.ravel(X_test_scaled @ beta)
            z_tilde_train[:, j] = np.ravel(X_train_scaled @ beta)
        elif method == funcs.calc_Ridge:
            beta = method(X_train_scaled_, z_train_scaled_, nlambda)
            z_tilde_test[:, j] = np.ravel(X_test_scaled @ beta)
            z_tilde_train[:, j] = np.ravel(X_train_scaled @ beta)
        elif method == "Lasso":
            clf = linear_model.Lasso(alpha = nlambda)
            clf.fit(X_train_scaled_, z_train_scaled_)
            z_tilde_test[:, j] = clf.predict(X_test_scaled)
            z_tilde_train[:, j] = clf.predict(X_train_scaled)



    MSE_bootstrap_test = np.mean(np.mean((z_test_scaled - z_tilde_test)**2,
                                            axis=1,
                                            keepdims=True))

    MSE_bootstrap_train = np.mean(np.mean((z_train_scaled - z_tilde_train)**2,
                                             axis=1,
                                             keepdims=True))

    bias_bootstrap = np.mean((z_test_scaled - np.mean(z_tilde_test,
                                                         axis=1,
                                                         keepdims=True))**2)

    variance_bootstrap = np.mean(np.var(z_tilde_test,
                                           axis=1,
                                           keepdims=True))

    data = {"polydegree":polydegree,
            "MSE_bootstrap_test":MSE_bootstrap_test,
            "MSE_bootstrap_train":MSE_bootstrap_train,
            "bias_bootstrap":bias_bootstrap,
            "variance_bootstrap":variance_bootstrap}


    return data


if __name__ == "__main__":
    """
    Doing some bootstrap analysis here 
    Most analysis for the methods is implemented in separat files. 
    
    """
    # Number of datapoints
    n = 20
    seed = 111
    
    filename = "SRTM_data_Norway_1.tif"

    x, y, z = funcs.make_data(n, seed=seed) #add filename as argument here for looking at terrain 

    # Degree
    degree = 20
    # Bootstrap degree
    nboot = 50
    method1 = funcs.calc_OLS
    method2 = funcs.calc_Ridge
    method3 = "Lasso"

    nlambda = 0.001

    #ols
    data = bootstrap_analysis(x, y, z, degree, nboot, method1)
    polydegree = data["polydegree"]
    MSE_bootstrap_test = data["MSE_bootstrap_test"]
    MSE_bootstrap_train = data["MSE_bootstrap_train"]
    bias_bootstrap = data["bias_bootstrap"]
    variance_bootstrap = data["variance_bootstrap"]
    
    funcs.plot_bootstrap(polydegree,
                         MSE_bootstrap_test,
                         MSE_bootstrap_train,
                         bias_bootstrap,
                         variance_bootstrap,
                        "OLS, bias-variance trade-off, n = " + str(n**2))
    
    
    
    #ridge
    data = bootstrap_analysis(x, y, z, degree, nboot, method2, nlambda)
    polydegree = data["polydegree"]
    MSE_bootstrap_test = data["MSE_bootstrap_test"]
    MSE_bootstrap_train = data["MSE_bootstrap_train"]
    bias_bootstrap = data["bias_bootstrap"]
    variance_bootstrap = data["variance_bootstrap"]
    
    funcs.plot_bootstrap(polydegree,
                         MSE_bootstrap_test,
                         MSE_bootstrap_train,
                         bias_bootstrap,
                         variance_bootstrap,
                        "Ridge, Bias-variance trade-off, n = " + str(n**2))
    
    

    """
    # Lasso
    data = bootstrap_analysis(x, y, z, degree, nboot, method3, nlambda)
    polydegree = data["polydegree"]
    MSE_bootstrap_test = data["MSE_bootstrap_test"]
    MSE_bootstrap_train = data["MSE_bootstrap_train"]
    bias_bootstrap = data["bias_bootstrap"]
    variance_bootstrap = data["variance_bootstrap"]

    funcs.plot_bootstrap(polydegree,
                         MSE_bootstrap_test,
                         MSE_bootstrap_train,
                         bias_bootstrap,
                         variance_bootstrap,
                        "Lasso, Bias-variance trade-off, n = " + str(n**2))
    """

    
    
    
    


