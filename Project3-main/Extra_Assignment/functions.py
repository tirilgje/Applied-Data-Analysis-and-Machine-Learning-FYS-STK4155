import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.utils import resample


from NN_keras import NN_Keras



def read_data(n_features=11):
    '''
    Reads the data, extract the features and the target into X and z,
    and splits the data into test and train data.
    (We do not scale the data becaise it is already done)

    Arguments:
        n_features (int): Number of features to use in the model
                          Must be an int between 1 and 11.

    Returns:
        X_train, X_test, z_train, z_test: Test and train data
    '''


    # Leser inn datasettet med sklearn
    X, z = load_diabetes(return_X_y=True)
    z = z.reshape((-1,1))
    X = X[:,:n_features]

    # Deler data inn i train og test
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)
    #print(X_train.shape, z_train.shape)

    # Skalerer ikke data siden det allerede er gjort i det vi leser inn fra sklearn

    return X_train, X_test, z_train, z_test


def MSE(z_data,z_pred):
    '''
    Calculates the mean squared error between the true data and the predicted data

    Arguments:
        z_data: The true data
        z_pred: The predicted data

    Returns:
        the mean squared error score
    '''
    n = np.size(z_pred)
    return np.sum((z_data-z_pred)**2)/n


def ols():
    '''
    Calculates the predicted output with ols

    Returns:
        z_tilde_ols: Predicted output
    '''

    beta = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ z_train)
    z_tilde_ols = X_test @ beta
    mse_ols = MSE(z_tilde_ols, z_test)

    #print('shape ols:', X_test.shape, beta.shape)

    return z_tilde_ols, mse_ols


def predict_regression(lmd=0):

    '''
    Predicts the output for both OLS and Ridge regression.
    In case of OLS, lmd = 0, in case og ridge, lmd > 0

    Arguments:
        lmd (float) [optional, default=0]: Regularization parameter lambda
                                           If lmd 0 -> OLS (default)
                                           If lmd > 0 -> Ridge

    Returns:
        pred (ndarray): Array of the predicted output
    '''
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]


    clf = Ridge(alpha=lmd, fit_intercept=False)
    clf.fit(X_train, z_train)



    pred = clf.predict(X_test)
    #print('shapes in predict')
    #print(pred.shape, X_test.shape)
    #score = clf.score(pred, z_test)
    #print(score)

    mse = MSE(pred, z_test)
    #print(mse)

    return pred, mse


def neural_network(eta, lmd, bsize, n_epochs, n_hidden, X_train, X_test, z_train, z_test):



    NN_keras = NN_Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, lmd, loss='mean_squared_error',
                 metrics=['mean_squared_error'], act_hidden='relu', act_out='sigmoid')

    model = NN_keras.NN_model()

    model.fit(X_train, z_train.reshape((-1,1)), epochs=n_epochs, batch_size=bsize, verbose=0)
    scores = model.evaluate(X_test, z_test.reshape((-1,1)))
    #print(f"MSE of test keras (Sigmoid) {scores[1]}")

    return scores[1]


def bootstrap(nboot, lmd, X_train, X_test, z_train, z_test):

    """
    The bootstrap algorithm we implemented in P1


    Arguments:
        nboot (int): Number of bootstrap samples
        lmd (float); Regularization parameter lambda. (lmd = 0 for ols, lmd > 0 for ridge)
        X_train, X_test, z_train, z_test: The training and test data
    """


    # Store  all the predicted z-values from each bootstrap
    z_tilde_test = np.zeros((np.shape(z_test)[0], nboot))
    z_tilde_train = np.zeros((np.shape(z_train)[0], nboot))

    index = np.arange(0, X_train.shape[0], 1)

    for j in range(nboot):
        id = resample(index) # resampling with replacement

        # Henter 1 random sample
        new_z_train = z_train[id]
        new_X_train = X_train[id, :]

        # Regner ut predikert verider
        clf = Ridge(alpha = lmd)                     # Ridge = OLS hvis lmd = 0
        clf.fit(new_X_train, new_z_train)            # Tilpasser modellen til sampelet

        #print('where it goes wrong')
        #print(X_test.shape, z_tilde_test[:,j].shape)

        z_test_pred = clf.predict(X_test)
        z_train_pred = clf.predict(X_train)

        #print(z_test_pred.shape, z_train_pred.shape)
        #print(z_test_pred.reshape((z_test_pred.shape[0])).shape)

        z_tilde_test[:,j] = z_test_pred.reshape((z_test_pred.shape[0]))    # Predikerer z_test (outputet)
        z_tilde_train[:,j] = z_train_pred.reshape((z_train_pred.shape[0])) # Predikerer z_train (outputet)




    # Calculates mse for test and train, after the bootstrap
    MSE_test = np.mean(np.mean((z_test - z_tilde_test)**2,
                                axis=1,
                                keepdims=True))

    MSE_train = np.mean(np.mean((z_train - z_tilde_train)**2,
                                 axis=1,
                                 keepdims=True))

    # Calculates the bias^2 and variance
    bias2 = np.mean((z_test - np.mean(z_tilde_test,
                                      axis=1,
                                      keepdims=True))**2)

    variance = np.mean(np.var(z_tilde_test,
                              axis=1,
                              keepdims=True))

    data = {"MSE_test":MSE_test,
            "MSE_train":MSE_train,
            "bias2":bias2,
            "variance":variance}


    return data




def bias_var_tradeoff_complexity(nboot, lmd, n_features):

    '''
    Bias-variance trade-off analysis for linear regression

    Arguments:
        nboot(int): Number of bootstraps
        n_lmds (int): Number of lambdas to test
    '''

    # Store MSE values for each lambda
    MSE_tests = np.zeros(n_features)

    # Store bias and variance
    biases2 = np.zeros(n_features)
    variances = np.zeros(n_features)



    for k in range(n_features):
        X_train, X_test, z_train, z_test = read_data(n_features = k+1)
        data  = bootstrap(nboot, lmd, X_train, X_test, z_train, z_test)

        biases2[k] = data['bias2']
        variances[k] = data['variance']
        MSE_tests[k] = data['MSE_test']



    #print(features.shape, biases2.shape)

    features = range(n_features)
    # Now plot the result
    plt.plot(features, biases2, label = r'$bias^2$')

    plt.plot(features, variances, label = 'Variance')
    plt.plot(features, biases2+variances, label = r'$bias^2+variance$')
    plt.plot(MSE_tests, '--', label = 'MSE')

    plt.xlabel('Number of features')
    plt.ylabel('Prediction Error')
    plt.title('Bias-variance trade-off')
    plt.yscale('log')
    plt.legend()
    plt.show()


def neural_network_complexity(n_features):

    MSEs = np.zeros(n_features)

    eta = 0.01
    lmd = 0.0001
    bsize = 16
    n_epochs = 20
    n_hidden = [50]

    for i in range(n_features):
        X_train, X_test, z_train, z_test = read_data(n_features = i+1)

        mse_NN = neural_network(eta, lmd, bsize, n_epochs, n_hidden, X_train, X_test, z_train, z_test)
        MSEs[i] = mse_NN

    plt.plot(range(n_features), MSEs)
    plt.xlabel('Number of features')
    plt.ylabel('Prediction Error')
    plt.title('Neural Network')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    return MSEs



if __name__ == '__main__':


    X_train, X_test, z_train, z_test = read_data()

    eta = 0.01
    lmd = 0.001
    bsize = 16
    n_epochs = 80
    n_hidden = [50]


    # Bare leker litt med metodene
    pred_ols, mse_ols = ols()
    pred_ols_sk, mse_ols_sk = predict_regression()
    pred_ridge, mse_ridge = predict_regression(lmd=lmd)
    mse_NN = neural_network(eta, lmd, bsize, n_epochs, n_hidden, X_train, X_test, z_train, z_test)

    print(mse_ols, mse_ols_sk, mse_ridge)

    nboot = 100      #change nboot here 
    lmd = 0          #change lambda here 
    n_features = 10
    n_lambdas = 10
    #bias_var_tradeoff_complexity(nboot, lmd, n_features)

    test_nn = neural_network_complexity(n_features)
    print(test_nn)
