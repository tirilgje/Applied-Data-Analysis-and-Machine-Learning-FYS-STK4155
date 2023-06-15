
import funcs

'''
Linear regression reused from project 1. 
'''

#Does general linear regression with out any resamplings techniques.

def linear_regression(x, y, z, order=5, lmd=0, method="calc_OLS"):
    """
    This function calculates the OLS of the inputs x, y and z created by the
    function make_data.

    z - FrankeFunction

    The function creates a design matrix X, and splits X and z into training data
    and testing data before scaling it. Then, the function find ztilde for both
    datasets, before finding the confidence interval for 95%. Next, it evaluates
    the MSE and R2 for both training and test data.

    The function returns two arrays containing different types of information.

    data: Contains all the different data, meaning the splitted X and z after scaling,
        and the test and train ztilde.
        data = [X_train_scaled, X_test_scaled, z_train_scaled, z_test_scaled, ztilde_scaled, z_tilde_scaled_test]
    r2mse: Contains all the values of the MSE and R2 for both test and train data.
        r2mse = [r2_train, r2_test, mse_train, mse_test]
    """
    X = funcs.create_X(x, y, n=5)
    data = funcs.prep_regression(X, z)

    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    z_train_scaled = data['z_train_scaled']
    z_test_scaled = data['z_test_scaled']


    # Calculating the OLS for the scaled data
    if method == "calc_OLS":
        beta_scaled = funcs.calc_OLS(X_train_scaled,
                               z_train_scaled)

    elif method == "calc_Ridge":
        beta_scaled = funcs.calc_Ridge(X_train_scaled,
                                 z_train_scaled,
                                 lmd)


    # Creates ztilde for the scaled data
    z_tilde = X_test_scaled @ beta_scaled

    # Evaluate MSE and R2 for the test data
    mse_test = funcs.MSE(z_test_scaled, z_test_scaled)
    r2_test = funcs.R2(z_test_scaled, z_test_scaled)

    # Creates arrays to contain all data wanted
    all_data = {"X_test_scaled":X_test_scaled,
                "z_test_scaled":z_test_scaled,
                "z_tilde":z_tilde}

    #return r2 and mse
    r2mse = {"r2_test":r2_test,
             "mse_test": mse_test}



    return all_data, mse_test
