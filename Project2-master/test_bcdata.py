import matplotlib.pyplot as plt
import numpy as np
import activation as act
import costfuncs as cost
from NN_kreas import Keras
from sklearn.linear_model import LogisticRegression
import seaborn as sb
from analysis_funcs import run_network_class, prep_network_bc, plot_results

'''
This is a testing file. Run this file to runs the tests done on the breast canser data. 
To run all the tests the variable run_all must be set to True.  


The file contains the following functions:
    plot_heatmap: Plots heatmap illustrating the results 
    test_epochs_bsize: Testing different values for epochs and batch size
    test_bsize_hidden: Testing different values for batch size and number of hidden layers
    test_eta_lmd: Testing different combinations of eta and lambda. 
'''

np.random.seed(2525)

#Set to True if you want to run all the tests. 
run_all = False

# Test the bc data on the neural network
X_train, X_test, z_train, z_test = prep_network_bc()


def plot_heatmap(ACC, xval, yval, xlab, ylab, title, filename=None):
    heatmap = sb.heatmap(ACC, annot=True, fmt='.4g', cmap='YlGnBu',
                     xticklabels=xval, yticklabels=yval,
                     cbar_kws={'label': 'Accuracy'})
    heatmap.set_ylabel(ylab, size=12)
    heatmap.set_xlabel(xlab, size=12)
    heatmap.invert_yaxis()
    heatmap.set_title(title, size=16)
    if filename:
        plt.savefig('figs_bc/' + filename)

def test_epochs_bsize(act_hidden, filename):

    n_hidden = [50]                       
    act_out = act.Activation_Sigmoid()
    eta = 1e-3
    my_cost = cost.CrossEntropy()

    epochs = range(50,501, 50)
    bsizes = [8,16,32,64, 128]
    ACCs = np.zeros((len(bsizes), len(epochs)))


    if isinstance(act_hidden, act.Activation_Sigmoid):
        title = "Accuracy of Sigmoid + Sigmoid, epochs vs batch size"
    elif isinstance(act_hidden, act.Activation_ReLU):
        title = "Accuracy of ReLU + Sigmoid, epochs vs batch size"
    elif isinstance(act_hidden, act.Activation_LeakyReLU):
        title = "Accuracy of Leaky ReLU + Sigmoid, epochs vs batch size"
    else: #logreg
        title = 'Logistic regression, epochs vs bsize'
        n_hidden = None


    plt.figure(figsize = [12, 8])

    print('Testing epochs vs bsize for NN with bc data ....')
    for i in range(len(epochs)):
        print(i+1, '/', len(epochs))
        for j in range(len(bsizes)):

  
            # Trains the networsk and tests the accuracy atfer 
            acc_train, acc_test = run_network_class(X_train, X_test, z_train, z_test, 
                                                    bsizes[j], epochs[j], eta, 
                                                    act_hidden, act_out, my_cost, n_hidden)

            
            ACCs[j,i] = acc_test[-1]


    plot_heatmap(ACCs, epochs, bsizes, "Epochs", 'Size of minibatches', title, filename=filename)
    plt.show()

def test_bsize_hidden(act_hidden, filename):


    if isinstance(act_hidden, act.Activation_Sigmoid):
        title = "Accuracy of Sigmoid + Sigmoid, hidden layers vs batch size"
    elif isinstance(act_hidden, act.Activation_ReLU):
        title = "Accuracy of ReLU + Sigmoid, hidden layers vs batch size"
    elif isinstance(act_hidden, act.Activation_LeakyReLU):
        title = "Accuracy of Leaky ReLU + Sigmoid, hidden layers vs batch size"

  
    act_out = act.Activation_Sigmoid()
    eta = 0.01
    my_cost = cost.CrossEntropy()
    epochs = 100

    n_hiddens = [[10], [50], [100], [10,10], [30,30], [50,50], [10,10,10],[15,15,15],[20,20,20]]
    bsizes = [8,16,32,64,128]
    ACCs = np.zeros((len(n_hiddens), len(bsizes)))
    plt.figure(figsize = [12, 8])

    print('Testing hidden layers vs bsize for NN with bc data ....')
    for i in range(len(n_hiddens)):
        print(i+1, '/', len(n_hiddens))
        for j in range(len(bsizes)):

            # Trains the networsk and tests the accuracy atfer 
            acc_train, acc_test = run_network_class(X_train, X_test, z_train, z_test, 
                                                    bsizes[j], epochs, eta, 
                                                    act_hidden, act_out, my_cost, n_hiddens[i])

            
            ACCs[i,j] = acc_test[-1]


    plot_heatmap(ACCs, bsizes, n_hiddens, 'Size of minibatches', 'n_hidden', title, filename=filename)
    plt.show()

def test_eta_lmd(act_hidden, filename):
    n_hidden = [50]                      
    act_out = act.Activation_Sigmoid()
    epochs = 150
    bsize = 64

    lmds = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    etas = [1e-4, 1e-3, 1e-2, 0.02, 0.03]

    my_cost = cost.CrossEntropy()

    ACCs = np.zeros((len(etas), len(lmds)))

    if isinstance(act_hidden, act.Activation_Sigmoid):
        title = "Accuracy of Sigmoid + Sigmoid, eta vs lmd"
    elif isinstance(act_hidden, act.Activation_ReLU):
        title = "Accuracy of ReLU + Sigmoid, eta vs lmd"
    elif isinstance(act_hidden, act.Activation_LeakyReLU):
        title = "Accuracy of Leaky ReLU + Sigmoid, eta vs lmd"
    else: #logreg
        title = 'Logistic regression, eta vs lambda'
        n_hidden = None

    plt.figure(figsize = [12, 8])

    print('Testing eta vs lmd for NN with bc data ....')
    for i in range(len(etas)):
        print(i+1, '/', len(etas))
        for j in range(len(lmds)):
            acc_train, acc_test = run_network_class(X_train, X_test, z_train, z_test, 
                                                    bsize, epochs, etas[i], 
                                                    act_hidden, act_out, my_cost, n_hidden, lmd=lmds[j])

            
            ACCs[i,j] = acc_test[-1]


    plot_heatmap(ACCs, lmds, etas, 'lambdas', 'Learning rate', title, filename=filename)
    plt.show()


if run_all:
    # Analysing number of epochs and bsize 
    test_epochs_bsize(act.Activation_Sigmoid(), 'ep_b_sig')
    test_epochs_bsize(act.Activation_ReLU(), 'ep_b_relu')
    test_epochs_bsize(act.Activation_LeakyReLU(), 'ep_b_lrelu')

    # Analysing hidden layers and batch size 
    test_bsize_hidden(act.Activation_Sigmoid(), 'hid_b_sig')
    test_bsize_hidden(act.Activation_ReLU(), 'hid_b_relu')
    test_bsize_hidden(act.Activation_LeakyReLU(), 'hid_b_lrelu')

    # Analysing eta and lambda 
    test_eta_lmd(act.Activation_Sigmoid(), 'eta_lmd_sig')
    test_eta_lmd(act.Activation_ReLU(), 'eta_lmd_relu')
    test_eta_lmd(act.Activation_LeakyReLU(), 'eta_lmd_lrelu')

    # Analysing logistic regression 
    test_epochs_bsize(None, 'logreg_epochs_bsize')
    test_eta_lmd(None, 'logreg_eta_lmd')



bsize = 32                              
n_epochs = 100                          
eta = 0.01                              
act_hidden = act.Activation_Sigmoid()   
act_out = act.Activation_Sigmoid()   
my_cost = cost.CrossEntropy()   
n_hidden = [50]                         

# Trains the networsk and tests the accuracy atfer 
accs_train, accs_sigmoid = run_network_class(X_train, X_test, z_train, z_test, 
                                             bsize, n_epochs, eta, 
                                             act.Activation_Sigmoid(), act_out, my_cost, n_hidden)


accs_train, accs_relu = run_network_class(X_train, X_test, z_train, z_test, 
                                             bsize, n_epochs, eta, 
                                             act.Activation_ReLU(), act_out, my_cost, n_hidden)

accs_train, accs_leaky = run_network_class(X_train, X_test, z_train, z_test, 
                                             bsize, n_epochs, eta, 
                                             act.Activation_LeakyReLU(), act_out, my_cost, n_hidden)


plot_results(accs_sigmoid, 'Sigmoid', 'Epochs', 'Accuracy', 'Activation functions for BC data', filename='figs_bc/all_acts_NN')
plot_results(accs_relu, 'ReLU', 'Epochs', 'Accuracy', 'Activation functions for BC data', filename='figs_bc/all_acts_NN')
plot_results(accs_leaky, 'Leaky ReLU', 'Epochs', 'Accuracy', 'Activation functions for BC data', filename='figs_bc/all_acts_NN')

plt.show()

#Logistic regression
act_hidden = None                 
n_hidden = None    

# Trains the networsk and tests the accuracy atfer 
accs_train, accs_test_log = run_network_class(X_train, X_test, z_train, z_test, 
                                              bsize, n_epochs, eta, 
                                              act_hidden, act_out, my_cost, n_hidden)



plot_results(accs_test_log, 'logistic', 'Epochs', 'Accuracy', 'Logistic vs neural', filename='figs_bc/logreg_alone')
plt.show()


#All together i the same plot 
plot_results(accs_sigmoid, 'Sigmoid', 'Epochs', 'Accuracy', 'Activation functions for BC data', filename='figs_bc/logreg_NN')
plot_results(accs_relu, 'ReLU', 'Epochs', 'Accuracy', 'Activation functions for BC data', filename='figs_bc/logreg_NN')
plot_results(accs_leaky, 'Leaky ReLU', 'Epochs', 'Accuracy', 'Activation functions for BC data', filename='figs_bc/logereg_NN')
plot_results(accs_test_log, 'logistic', 'Epochs', 'Accuracy', 'Logistic regression vs Neural Network', filename='figs_bc/logreg_NN')
plt.show()




n_hidden = [50]

#Sigmoid
NN_keras = Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, 0, loss='binary_crossentropy',
                 metrics=['binary_accuracy'], act_hidden='sigmoid', act_out='sigmoid')

model = NN_keras.NN_model()

model.fit(X_train, z_train.reshape((-1,1)), epochs=n_epochs, batch_size=bsize, verbose=0)
scores = model.evaluate(X_test, z_test.reshape((-1,1)))
print(f"Accuracy test keras (sigmoid) {scores[1]}")


#ReLU
NN_keras = Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, 0, loss='binary_crossentropy',
                 metrics=['binary_accuracy'], act_hidden='relu', act_out='sigmoid')

model = NN_keras.NN_model()

model.fit(X_train, z_train.reshape((-1,1)), epochs=n_epochs, batch_size=bsize, verbose=0)
scores = model.evaluate(X_test, z_test.reshape((-1,1)))
print(f"Accuracy test keras (ReLU) {scores[1]}")



# Logistic Regression sklearn
logreg = LogisticRegression()
logreg.fit(X_train, z_train)
print("Test set accuracy with Logistic Regression {:.2f}".format(logreg.score(X_test, z_test)))

