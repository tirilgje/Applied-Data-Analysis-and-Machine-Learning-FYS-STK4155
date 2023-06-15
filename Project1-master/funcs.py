from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from imageio import imread

def make_data(n, seed = 3843, filename = None):
    """
    This function makes the data we are interested in. It takes no input, but sends
    out the three outputs x, y and z.

    x, y : arrays from 0 to 1 with 20 steps.
    z    : FrankeFunction with noise.

    """
    if filename is None:
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        x, y = np.meshgrid(x,y)

        # To keep the random numbers "the same"
        np.random.seed(seed)

        # Creates z from the FrankeFunction, but adds some random noise.
        z = FrankeFunction(x,y) + 0.1 * np.random.randn(x.shape[0], x.shape[1])
    else:
        terrain = imread(filename)
        terrain_down = down_sample(terrain, 80)

        # Fixing a set of points
        terrain_scaled = terrain_down[:n, :n]
        print(terrain_scaled.shape)
        
        show_terrain(terrain_scaled, 'Terrain over Norway 1', 'X', 'Y')
        show_terrain(terrain_down, 'Terrain over Norway 1', 'X', 'Y')

        
        

        # Creates mesh of image pixels
        x = np.linspace(0,1, np.shape(terrain_scaled)[0])
        y = np.linspace(0,1, np.shape(terrain_scaled)[1])
        x, y = np.meshgrid(x,y)

        z = terrain_scaled

    # plot_franke(x, y, z)
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()

    return x, y, z


def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def plot_franke(x, y, z):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z,
                           cmap=cm.coolwarm,
                           linewidth=0,
                           antialiased=False)

    # Customize the z axis.
    #ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.zlabel("z")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5, label = "z")
    plt.legend()
    plt.show()



def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n


def create_X(x, y, n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)     # Number of elements in beta
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)

    return X

def calc_OLS(X, z):
    return np.linalg.pinv(X.T @ X) @ X.T @ z

def calc_Ridge(X, z, lmb):

    """
    Agruments:
        X (numpy ndarray): designmatrix X
        z(numpy ndarray): predicted z matrix
        lmb(float):
    Returns:
        beta
    """

    p = X.shape[1]
    I = np.eye(p,p)

    return np.linalg.pinv(X.T @ X+lmb*I) @ X.T @ z


def prep_linear_regression(x, y, z, order=5):

    # Create the design matrix
    X = create_X(x, y, n=order)
    

    # Split into test and train data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)


    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    scaler2 = StandardScaler()
    scaler2.fit(z_train.reshape(-1,1))
    z_train_scaled = scaler2.transform(z_train.reshape(-1,1))
    z_test_scaled = scaler2.transform(z_test.reshape(-1,1))
    
    
    # Creates arrays to contain all data wanted
    data = {"X_train_scaled":X_train_scaled,
            "X_test_scaled":X_test_scaled,
            "z_train_scaled":z_train_scaled,
            "z_test_scaled":z_test_scaled}
    
    
    return data 



def down_sample(terrain_data, N):
    """
    Down sample terrain data
    """
    m, n = terrain_data.shape
    
    m_down, n_down = int(m / N), int(n / N)
    
    terrain_down_sample = np.zeros((m_down, n_down))
    
    for i in range(m_down):
        for j in range(n_down):
            slice = terrain_data[N * i:N * (i + 1), N * j:N * (j + 1)]
            terrain_down_sample[i, j] = np.mean(slice)
            
    return terrain_down_sample

def show_terrain(data, title, xlabel, ylabel):
    
    # Show the terrain
    plt.figure()
    plt.title(title)
    plt.imshow(data, cmap="gray")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()




def plot_bootstrap(polydegree, MSE_bootstrap_test, MSE_bootstrap_train, bias_bootstrap, variance_bootstrap, title):
    #Test cv train:
    #plt.plot(polydegree, MSE_bootstrap_test, label='MSE test')
    #plt.plot(polydegree, MSE_bootstrap_train, label='MSE train')

    #bias-variance analysis
    plt.plot(polydegree, bias_bootstrap, label='bias^2')
    plt.plot(polydegree, variance_bootstrap, label='variance')
    plt.plot(polydegree, bias_bootstrap+variance_bootstrap, label='bias^2+variance')
    plt.plot(polydegree, MSE_bootstrap_test, '--', label = 'MSE test')
    plt.xlabel("Model complexity (polynomial degree)")
    plt.ylabel("Prediction error")
    plt.yscale("log")
    plt.title(title)
    plt.legend()
    plt.show()



def plot_ols(polydegree, mse_test, mse_train, title):

    plt.plot(polydegree, mse_test, label='MSE test')
    plt.plot(polydegree, mse_train, label='MSE train')
    plt.xlabel("Model complexity (polynomial degree)")
    plt.ylabel("Prediction error")
    plt.title(title)
    plt.yscale("log")
    plt.legend()
    plt.show()

def plot_cross_validation(polydegree, MSE_train, MSE_test, title, method):
    if method == "Lasso":
        plt.figure()
        plt.title(title)
        plt.plot(polydegree, MSE_train, label = "MSE train")
        plt.plot(polydegree, MSE_test, label = "MSE test")
        plt.xlabel("Model complexity (polynomial degree)")
        plt.ylabel('Prediction Error')
        plt.legend()
        plt.show()
    else:
        #plt.figure()
        #plt.subplot(121)
        plt.title(title)
        plt.plot(polydegree, MSE_train, label = "MSE Train")
        plt.plot(polydegree, MSE_test, label = "MSE Test")
        plt.xlabel("Model complexity (polynomial degree)")
        plt.ylabel('Prediction Error')
        plt.legend()
        #plt.subplot(122)
        #plt.title("Sklearn")
        #plt.plot(polydegree, MSE_mean_skl, label = "MSE")
        #plt.plot(polydegree, R2_skl, label = "R2")
        #plt.legend()
        #plt.suptitle(title)
        plt.show()


def confidence_interval(X_train_scaled,z_train_scaled):

    # Begins with the confidence interval
    # n for confidence intervall
    n_conf = z_train_scaled.shape[0]

    #sigma^2 (en skalar!)
    s_2 = np.var(z_train_scaled)

    #cov(beta) skal bruke diagonalen
    cov_beta = s_2 * np.linalg.pinv(X_train_scaled.T @ X_train_scaled)
    #hente ut diagnonalen:
    cov_beta = np.diag(cov_beta)

    beta_scaled = calc_OLS(X_train_scaled, z_train_scaled)
    
    CI = (1.96 * np.sqrt(cov_beta)) / np.sqrt(n_conf)


    lower = beta_scaled - CI
    upper = beta_scaled + CI
    

    confidenceinterval = list(zip(lower, upper))
    
    
    #plot results, i tried men ser ikke bra ut://
    plt.errorbar(range(len(beta_scaled)), beta_scaled, CI, fmt="b.", capsize=3, label=r'$\beta_j \pm 1.96 \sigma$')
    #plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.legend()
    plt.xlabel(r'index $j$')
    plt.ylabel(r'$\beta_j$')
    plt.grid()
    plt.show()

    
    return confidenceinterval
