import funcs
import matplotlib.pyplot as plt
import numpy as np
from SGD import StocasticGradientDescent
import seaborn as sb
from sklearn.linear_model import SGDRegressor


'''
This is a testing file. Run this file to runs the tests done on the the franke data with SGD.
To run all the tests the variable run_all must be set to True.  

The file contains the following functions functions testing all the parameters
'''

#Set to True if you eant to run all the tests
run_all = False

#Create the data
x, y, z = funcs.make_data(20)
X = funcs.create_X(x, y, n=5)
data = funcs.prep_regression(X, z)

X_train = data["X_train_scaled"]
X_test = data["X_test_scaled"]
z_train = data["z_train_scaled"]
z_test = data["z_test_scaled"]

sgdClass = StocasticGradientDescent(X_train, X_test, z_train,z_test)

def plot_heatmap(MSE, xval, yval, xlab, ylab, title, filename=None):
    heatmap = sb.heatmap(MSE, annot=True, fmt='.4g', cmap='YlGnBu',
                     xticklabels=xval, yticklabels=yval,
                     cbar_kws={'label': 'MSE'})
    heatmap.set_ylabel(ylab, size=12)
    heatmap.set_xlabel(xlab, size=12)
    heatmap.invert_yaxis()
    heatmap.set_title(title, size=16)
    if filename:
        plt.savefig('figs_partA/' + filename)
    # plt.show()

def testing_epochsvsbatches_OLS():
    eta = 0.01
    gamma = 0.6

    bsizes = [4,8,16,32]
    epochs = [500, 750, 1000, 1250, 1500, 1750, 2000]

    MSEs = np.zeros((len(epochs), len(bsizes)))

    print('Testing epochs vs batches for OLS ....')
    for i in range(len(epochs)):
        print(i+1, '/', len(epochs))
        for j in range(len(bsizes)):
            theta = sgdClass.SGD(epochs[i], bsizes[j], gamma=gamma, _eta=eta)
            MSEs[i,j] = sgdClass.mse[-1]

    plot_heatmap(MSEs, bsizes, epochs, 'Size of minibatches', 'Epochs', 'MSE of Epochs vs minibatches, OLS', filename='epoch_nbatch_ols')
    plt.show()


def testing_epochsvsbatches_Ridge():
    lmb = 1e-5
    eta = 0.01
    gamma = 0.6

    bsizes = [4,8,16,32]
    epochs = [500, 750, 1000, 1250, 1500, 1750, 2000]

    MSEs = np.zeros((len(epochs), len(bsizes)))

    print('Testing epochs vs batches for Ridge ....')
    for i in range(len(epochs)):
        print(i+1, '/', len(epochs))
        for j in range(len(bsizes)):
            theta = sgdClass.SGD(epochs[i], bsizes[j], gamma=gamma, _eta=eta, lmb=lmb)
            MSEs[i,j] = sgdClass.mse[-1]

    plot_heatmap(MSEs, bsizes, epochs, 'Size of minibatches', 'Epochs', 'MSE of Epochs vs minibatches, Ridge', filename='epoch_nbatch_ridge')
    plt.show()


def testing_etavsbatches_OLS():
    gamma = 0.6
    epochs = 1500

    bsizes = [4,8,16,32, 64]
    etas = [1e-5, 1e-4, 1e-3, 1e-2, 0.02, 0.03, 0.04, 0.05]

    MSEs = np.zeros((len(etas), len(bsizes)))

    print('Testing eta vs batches for OLS ....')
    for i in range(len(etas)):
        print(i+1, '/' , len(etas))
        for j in range(len(bsizes)):
            theta = sgdClass.SGD(epochs, bsizes[j], gamma=gamma, _eta=etas[i])
            MSEs[i,j] = sgdClass.mse[-1]
            if MSEs[i,j] > 5:
                MSEs[i,j] = None

    plot_heatmap(MSEs, bsizes, etas, 'Size of minibatches', 'Eta', 'MSE of Eta vs minibatches, OLS', filename='eta_nbatch_ols')
    plt.show()


def testing_etaVSgamma_OLS():
    epochs = 1500
    bsize = 8

    gammas = np.linspace(0.2,0.7, 6) # Disse verdiene for å unngå overflow og cocoloco
    etas = [1e-5, 1e-4, 1e-3, 1e-2, 0.02, 0.03]

    MSEs = np.zeros((len(gammas), len(etas)))

    print('Testing eta vs gamma for OLS ....')
    for i in range(len(gammas)):
        print(i+1, '/', len(gammas))
        for j in range(len(etas)):
            theta = sgdClass.SGD(epochs, bsize, gamma=gammas[i], _eta=etas[j])
            MSEs[i,j] = sgdClass.mse[-1]

    plot_heatmap(MSEs, etas, gammas, 'Eta', 'Gamma', 'MSE of Etas vs Gammas, OLS', filename='eta_gamma_ols')
    plt.show()

def testing_etaVSlambda_Ridge():
    # Beste verdier fra testing_epochsvsbatches_OLS :)
    epochs = 1500
    bsize = 8

    gamma = 0.6
    lmbs = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1]
    etas = [1e-5, 1e-4, 1e-3, 1e-2, 0.02, 0.03]

    MSEs = np.zeros((len(lmbs), len(etas)))

    print('Testing eta vs lambda for Ridge ....')
    for i in range(len(lmbs)):
        print(i+1, '/', len(lmbs))
        for j in range(len(etas)):
            theta = sgdClass.SGD(epochs, bsize, gamma = gamma, _eta=etas[j], lmb = lmbs[i])
            MSEs[i,j] = sgdClass.mse[-1]

    plot_heatmap(MSEs, etas, lmbs, 'Eta', 'Lambdas', 'MSE of Etas vs Lambda, Ridge', filename='eta_lmd_ridge')
    plt.show()

def testing_gammaVSlambda_Ridge():
    # Beste verdier fra testing_epochsvsbatches_OLS :)
    epochs = 1500
    bsize = 8

    eta = 0.01
    lmbs = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1]
    gammas = np.linspace(0.2,0.7, 6) # Disse verdiene for å unngå overflow og cocoloco

    MSEs = np.zeros((len(lmbs), len(gammas)))

    print('Testing gamma vs lambda for Ridge ....')
    for i in range(len(lmbs)):
        print(i+1, '/', len(lmbs))
        for j in range(len(gammas)):
            theta = sgdClass.SGD(epochs, bsize, gamma = gammas[j], _eta=eta, lmb = lmbs[i])
            MSEs[i,j] = sgdClass.mse[-1]

    plot_heatmap(MSEs, gammas, lmbs, 'Gammas', 'Lambdas', 'MSE of Gammas vs Lambda, Ridge', filename='gamma_lmd_ridge')
    plt.show()


def testing_gammaVSeta_Ridge():
    # Beste verdier fra testing_epochsvsbatches_OLS :)
    epochs = 1500
    bsize = 8

    lmbs = [1e-5, 1e-4, 1e-3, 1e-2]
    etas = [1e-5, 1e-4, 1e-3, 1e-2, 0.02, 0.03]
    gammas = np.linspace(0.2,0.7, 6) # Disse verdiene for å unngå overflow og cocoloco

    plt.figure(figsize = [15, 10])

    print('Testing gamma vs eta vs lambda for Ridge ....')
    for k in range(len(lmbs)):
        print('k = ', k+1, '/', len(lmbs))
        MSEs = np.zeros((len(etas), len(gammas)))

        for i in range(len(etas)):
            for j in range(len(gammas)):
                theta = sgdClass.SGD(epochs, bsize, gamma = gammas[j], _eta=etas[i], lmb = lmbs[k])
                MSEs[i,j] = sgdClass.mse[-1]
        plt.subplot(2,2,1+k)
        plot_heatmap(MSEs, gammas, etas, 'Gammas', 'Etas', 'MSE of Gammas vs Eta, Ridge', filename='gamma_eta_lmd_ridge')

    plt.show()

def testing_gammaVSlambda_Ridge_dyneta():
    # Beste verdier fra testing_epochsvsbatches_OLS :)
    epochs = 1500
    bsize = 8

    lmbs = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1]
    gammas = np.linspace(0.2,0.7, 6) # Disse verdiene for å unngå overflow og cocoloco

    MSEs = np.zeros((len(lmbs), len(gammas)))

    print('Testing gamma vs lambda for Ridge with dyn eta ....')
    for i in range(len(lmbs)):
        print(i+1, '/', len(lmbs))
        for j in range(len(gammas)):
            theta = sgdClass.SGD(epochs, bsize, t0=20, t1=20, gamma = gammas[j], lmb = lmbs[i])
            MSEs[i,j] = sgdClass.mse[-1]

    plot_heatmap(MSEs, gammas, lmbs, 'Gammas', 'Lambdas', 'MSE of Gammas vs Lambda, Ridge', filename='gamma_lmd_dyn_ridge')
    plt.show()

def testing_t0t1_Ridge_dyneta():
    # Beste verdier fra tidligere testing :)
    epochs = 1500
    bsize = 8

    gamma = 0.6
    lmb = 0.01

    t0 = np.arange(5,21)
    t1 = np.arange(20,80,5)

    MSEs = np.zeros((len(t0), len(t1)))
    plt.figure(figsize = [15, 10])


    print('Testing startvalues for dyn eta for OLS ....')
    for i in range(len(t0)):
        print(i+1, '/', len(t0))
        for j in range(len(t1)):
            theta = sgdClass.SGD(epochs, bsize, t0 = t0[i], t1 = t1[j], gamma = gamma, lmb = lmb)
            MSEs[i,j] = sgdClass.mse[-1]

    plot_heatmap(MSEs, t1, t0, 't1', 't0', 'MSE of Gammas vs Lambda, Ridge', filename='t0_t1_ridge')
    plt.show()


if run_all:
    # Tester OLS parametere
    testing_epochsvsbatches_OLS()
    testing_etaVSgamma_OLS()
    testing_etavsbatches_OLS()

    # Tester div Ridge parametere
    testing_etaVSlambda_Ridge()
    testing_gammaVSlambda_Ridge()
    testing_gammaVSeta_Ridge()
    testing_gammaVSlambda_Ridge_dyneta()
    testing_epochsvsbatches_Ridge()

    # Test på t0 vs t1
    testing_t0t1_Ridge_dyneta()



# Testing av "Beste parametre"
# Finner optimal beta og tilhørende mse
beta = np.linalg.pinv(X_train.T @ X_train) @ (X_train.T @ z_train)
z_tilde_ols = X_test @ beta
mse_ols = funcs.MSE(z_tilde_ols, z_test)
print("MSE OLS: ", mse_ols)

print("Verdier eta=0.01, gamma = 0.6, lmbd = 1e-5:")
# Sjekker beste MSE for våre testede beste parametere
theta = sgdClass.SGD(1500, 8, _eta=0.01, gamma=0.6, lmb=1e-5)
print("     Testing MSE SGD, epochs = 1500:", sgdClass.mse[-1])

# z_tilde = X_test @ theta
# mse = funcs.MSE(z_tilde, z_test)
# print("     Testing MSE SGD, epochs = 1500:", mse)

# MSE for beste parametere, men større epochs. Diskusjon her
theta = sgdClass.SGD(3000, 8, _eta=0.01, gamma=0.6, lmb=1e-5)
print("     Testing MSE SGD, epochs = 3000:", sgdClass.mse[-1])


# z_tilde = X_test @ theta
# mse = funcs.MSE(z_tilde, z_test)
# print("     Testing MSE SGD, epochs = 3000:", mse)

print("Verdier dynamic eta, gamma = 0.6, lmbd = 0.01:")
# MSE for dynamisk eta, ikke like bra
theta = sgdClass.SGD(1500, 8, t0=20,t1=20, gamma=0.6, lmb=0.01)
print("     Testing MSE SGD, epochs = 1500:", sgdClass.mse[-1])

# z_tilde = X_test @ theta
# mse = funcs.MSE(z_tilde, z_test)
# print("     Testing MSE SGD, epochs = 1500:", mse)


theta = sgdClass.SGD(3000, 8, t0=20,t1=20, gamma=0.6, lmb=0.01)
print("     Testing MSE SGD, epochs = 3000:", sgdClass.mse[-1])

# z_tilde = X_test @ theta
# mse = funcs.MSE(z_tilde, z_test)
# print("     Testing MSE SGD, epochs = 3000:", mse)

"""
Midlertidig beste verdier:
    epochs = 1500       ->      Tradeoff, velger denne fordi mindre utregninger
    bsize = 8

    gamma = 0.6
    eta = 0.02 ( Har også dynamisk )
    lambda = 1e-5 -> OLS  // lambda = 0.01 ved dynamisk eta
"""



#sklearn implementation heheh funker ikke serlig bra idk hva som skjer :)

sgdreg = SGDRegressor(max_iter=1500, eta0=0.02, penalty=None, fit_intercept=False, tol=1e-10, learning_rate='constant')
sgdreg.fit(X_train, z_train.ravel())
#print("sgdreg from scikit")
#print(sgdreg.intercept_, sgdreg.coef_)
z_sklearn = X_test @ sgdreg.coef_.reshape(beta.shape)
mse_sklearn = funcs.MSE(z_sklearn, z_test)

print('mse skleran:', mse_sklearn)
