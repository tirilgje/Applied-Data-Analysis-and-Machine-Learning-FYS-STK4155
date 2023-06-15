import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
This file cotains different functions used in the code. Reused from projekt 1. 
'''

def make_data(n, seed = 3843):
    """
    This function makes the data we are interested in. It takes the number of
    datapoints, n, as input, and returns the three outputs x, y and z.

    x, y : arrays from 0 to 1 with 20 steps.
    z    : FrankeFunction with noise.
    """
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    x, y = np.meshgrid(x,y)

    # To keep the random numbers "the same"
    np.random.seed(seed)

    # Creates z from the FrankeFunction, but adds some random noise.
    z = FrankeFunction(x,y) + 0.1 * np.random.randn(x.shape[0], x.shape[1])



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


def Accuracy(y_data, y_model):
    return np.mean(y_data == y_model)



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


def prep_regression(X, z, order=5):

    # Create the design matrix
    # X = create_X(x, y, n=order)


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


#code from lecture week 40
def calc_theta_SGD(X, y, n_epochs, M, t0=None, t1=None, eta=None, theta=None):

    """
    Arguments:
        X: design matrix, y: predicted values
        n: number of datapoints, n_epochs: number of epochs
        M: size of each minibatch
        t0, t1: learning stuff dynamic learning rate
        eta: constant learning rate
    """

    m = int(n/M) #number of minibatches


    def learning_schedule(t):
        return t0/(t+t1)

    if theta is None:
        theta = np.random.randn(2,1)

    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X[random_index:random_index+1]
            yi = y[random_index:random_index+1]
            gradients = 2.0* xi.T @ ((xi @ theta)-yi)

            if eta is None:
                eta = learning_schedule(epoch*m+i)

            theta = theta - eta*gradients
    print("theta from own sdg")
    print(theta)

    return theta

# equivalent in numpy
def accuracy_score_numpy(z_test, z_pred):
    return np.sum(z_test == z_pred) / len(z_test)
