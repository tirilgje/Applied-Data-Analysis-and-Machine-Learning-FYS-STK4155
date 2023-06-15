from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl


import funcs
import linear_regression



def cross_validation(x, y, z, n, maxdegree, n_folds, seed, method, nlambda=0):
    if (n**2)%n_folds != 0:
        raise ValueError("Please let number of datapoints and number of folds be divisible by each others.")

    # Creates array so contain everything
    # _skl is for comparing with sklearn
    polydegree = np.zeros(maxdegree)
    MSE_mean_train = np.zeros(maxdegree)
    MSE_test = np.zeros(maxdegree)
    MSE_mean_skl = np.zeros(maxdegree)
    #R2_mean = np.zeros(maxdegree)
    #R2_skl = np.zeros(maxdegree)

    #calculation for all degrees up to maxdegree
    for deg in range(maxdegree):
        data = funcs.prep_linear_regression(x, y, z, deg+1)

        X_train_scaled = data["X_train_scaled"]
        X_test_scaled = data["X_test_scaled"]
        z_train_scaled = data["z_train_scaled"]
        z_test_scaled = data["z_test_scaled"]

        polydegree[deg] = deg+1

        # Shuffles the data
        index = np.arange(0, np.shape(X_train_scaled)[0], 1)
        np.random.seed(int(seed))
        np.random.shuffle(index)
        Xrandom = X_train_scaled[index,:]
        zrandom = z_train_scaled[index,:]


        # Splits the data
        kfold_X_train = np.array(np.array_split(Xrandom, n_folds))
        kfold_z_train = np.array(np.array_split(zrandom, n_folds))

        if method == funcs.calc_OLS:
            # Trying out the sklearn part for the OLS
            met = skl.LinearRegression(fit_intercept=False).fit(Xrandom, zrandom)
            scoreskl = cross_val_score(met, Xrandom, zrandom, cv = n_folds, scoring = 'neg_mean_squared_error')
            MSE_mean_skl[deg] = np.abs(np.mean(scoreskl))
            #R2_skl[deg] = np.mean(cross_val_score(met, Xrandom, zrandom, scoring='r2', cv=n_folds))
        if method == funcs.calc_Ridge:
            # Trying out the sklearn part for the Ridge regression method
            met = skl.Ridge(alpha=nlambda, fit_intercept = False).fit(Xrandom, zrandom)
            scoreskl = cross_val_score(met, Xrandom, zrandom, cv = n_folds, scoring = 'neg_mean_squared_error')
            MSE_mean_skl[deg] = np.abs(np.mean(scoreskl))
            #R2_skl[deg] = np.mean(cross_val_score(met, Xrandom, zrandom, scoring='r2', cv=n_folds))

        if method == "Lasso":
            # Implements the Lasso cross validaion
            met = skl.Lasso(alpha = nlambda, fit_intercept = False).fit(Xrandom, zrandom)
            scoreskl = cross_val_score(met, Xrandom, zrandom, cv = n_folds, scoring = 'neg_mean_squared_error')
            MSE_mean_skl[deg] = np.abs(np.mean(scoreskl))
            #R2_skl[deg] = np.mean(cross_val_score(met, Xrandom, zrandom, scoring='r2', cv=n_folds))

        if method != "Lasso":
            for k in range(n_folds):
                # Gathers the current split data
                X_k = kfold_X_train[k]
                z_k = np.reshape(kfold_z_train[k], (-1, 1))

                # Training data
                idx = np.ones(n_folds, dtype=bool)
                idx[k] = False
                X_train_fold = kfold_X_train[idx]

                # Combine folds
                X_train_fold = np.reshape(X_train_fold, (X_train_fold.shape[0]*X_train_fold.shape[1], X_train_fold.shape[2]))
                z_train_fold = np.reshape(np.ravel(kfold_z_train[idx]), (-1, 1))

                if method == funcs.calc_OLS:
                    # Calculates the OLS
                    beta_fold = method(X_train_fold, z_train_fold)

                elif method == funcs.calc_Ridge:
                    beta_fold = method(X_train_fold, z_train_fold, nlambda)

                z_tilde_fold = X_k @ beta_fold

                # Gathers all MSE and R2's
                MSE_mean_train[deg] += funcs.MSE(z_k, z_tilde_fold)
                #R2_mean[deg] += funcs.R2(z_k, z_tilde_fold)



            # Calculate the mean of the MSE and R2
            MSE_mean_train[deg] = MSE_mean_train[deg]/n_folds
            #R2_mean[deg] = R2_mean[deg]/n_folds

            z_tilde_test = X_test_scaled @ beta_fold

            MSE_test[deg] = funcs.MSE(z_tilde_test, z_test_scaled)



    data = {"polydegree":polydegree,
            "MSE_mean_train":MSE_mean_train,
            "MSE_test":MSE_test,
            "MSE_mean_skl":MSE_mean_skl}


    return data

def cross_validation_1run(x, y, z, n, polydegree, n_folds, seed, method, nlambda):
    """
    same algorithm as cross_validartion, bus only one run. Not for various complexities
    
    """
    
    
    if (n**2)%n_folds != 0:
        raise ValueError("Please let number of datapoints and number of folds be divisible by each others.")

    data = funcs.prep_linear_regression(x, y, z, polydegree)

    X_train_scaled = data["X_train_scaled"]
    X_test_scaled = data["X_test_scaled"]
    z_train_scaled = data["z_train_scaled"]
    z_test_scaled = data["z_test_scaled"]

    # Shuffles the data
    index = np.arange(0, np.shape(X_train_scaled)[0], 1)
    np.random.seed(int(seed))
    np.random.shuffle(index)
    Xrandom = X_train_scaled[index,:]
    zrandom = z_train_scaled[index,:]


    # Splits the data
    kfold_X_train = np.array(np.array_split(Xrandom, n_folds))
    kfold_z_train = np.array(np.array_split(zrandom, n_folds))

    if method == funcs.calc_OLS:
        # Trying out the sklearn part for the OLS
        met = skl.LinearRegression(fit_intercept=False).fit(Xrandom, zrandom)
        scoreskl = cross_val_score(met, Xrandom, zrandom, cv = n_folds, scoring = 'neg_mean_squared_error')
        MSE_mean_skl = np.abs(np.mean(scoreskl))

    if method == funcs.calc_Ridge:
        # Trying out the sklearn part for the Ridge regression method
        met = skl.Ridge(alpha=nlambda, fit_intercept = False).fit(Xrandom, zrandom)
        scoreskl = cross_val_score(met, Xrandom, zrandom, cv = n_folds, scoring = 'neg_mean_squared_error')
        MSE_mean_skl = np.abs(np.mean(scoreskl))

    if method == 'Lasso':
        # Implements the Lasso cross validaion
        met = skl.Lasso(alpha = nlambda, fit_intercept = False)
        met.fit(Xrandom, zrandom)
        scoreskl = cross_val_score(met, Xrandom, zrandom, cv = n_folds, scoring = 'neg_mean_squared_error')
        MSE_mean_skl = np.abs(np.mean(scoreskl))

        MSE_mean_train = np.abs(np.mean(scoreskl))
        z_tilde_test = met.predict(X_test_scaled)
        MSE_test = funcs.MSE(z_test_scaled, z_tilde_test)


    if method != 'Lasso':
        for k in range(n_folds):
            # Gathers the current split data
            X_k = kfold_X_train[k]
            z_k = np.reshape(kfold_z_train[k], (-1, 1))

            # Training data
            idx = np.ones(n_folds, dtype=bool)
            idx[k] = False
            X_train_fold = kfold_X_train[idx]

            # Combine folds
            X_train_fold = np.reshape(X_train_fold, (X_train_fold.shape[0]*X_train_fold.shape[1], X_train_fold.shape[2]))
            z_train_fold = np.reshape(np.ravel(kfold_z_train[idx]), (-1, 1))

            if method == funcs.calc_OLS:
                # Calculates the OLS
                beta_fold = method(X_train_fold, z_train_fold)

            elif method == funcs.calc_Ridge:
                beta_fold = method(X_train_fold, z_train_fold, nlambda)

            z_tilde_fold = X_k @ beta_fold

            # Gathers all MSE
            MSE_mean_train = funcs.MSE(z_k, z_tilde_fold)

        # Calculate the mean of the MSE
        MSE_mean_train = MSE_mean_train/n_folds

        z_tilde_test = X_test_scaled @ beta_fold

        MSE_test = funcs.MSE(z_tilde_test, z_test_scaled)


    data = {"polydegree":polydegree,
            "MSE_mean_train":MSE_mean_train,
            "MSE_test":MSE_test,
            "MSE_mean_skl":MSE_mean_skl}

    return data

if __name__ == '__main__':
    seed = 1345
    maxdegree = 20
    n_folds = 5     # number of folds
    n = 20 # Number of datapoints

    filename = "SRTM_data_Norway_1.tif"

    x, y, z = funcs.make_data(n, seed) #add filename as argument to look at terrain data 

    method1 = funcs.calc_OLS
    method2 = funcs.calc_Ridge
    method3 = 'Lasso'

    nlambda = 0.001
    #lambda i intervall log(-12) - log(-1) ish

    # # OLS
    data  = cross_validation(x, y, z, n, maxdegree, n_folds, seed, method1)
    
    polydegree = data["polydegree"]
    MSE_mean_train = data["MSE_mean_train"]
    MSE_test = data["MSE_test"]
    
    funcs.plot_cross_validation(polydegree,
                                MSE_mean_train,
                                MSE_test,
                                "Cross validation for OLS",
                                method1)





    # Ridge

    data  = cross_validation(x, y, z, n, maxdegree, n_folds, seed, method2, nlambda)

    polydegree = data["polydegree"]
    MSE_train = data["MSE_mean_train"]
    MSE_test = data["MSE_test"]

    funcs.plot_cross_validation(polydegree,
                                MSE_train,
                                MSE_test,
                                "Cross-validation for Ridge Regression",
                                method2)

    # Lasso
    # data  = cross_validation(x, y, z, n, maxdegree, n_folds, seed, method3, nlambda=0.05)
    #
    # polydegree = data["polydegree"]
    # MSE_train = data["MSE_mean_train"]
    # MSE_test = data["MSE_test"]
    #
    #
    #
    # funcs.plot_cross_validation(polydegree,
    #                             MSE_train,
    #                             MSE_test,
    #                             "Lasso",
    #                             method3)
