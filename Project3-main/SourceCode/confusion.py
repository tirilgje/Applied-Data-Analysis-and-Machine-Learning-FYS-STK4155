import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from FruitReads import read_fruit
from NN_keras import NN_Keras
from CNN_keras import CNN_Keras
import activation as act
import costfuncs as cost
from NeuralNetwork import NeuralNetwork
import funcs
from test_NN import train_network_class

'''
This file contains a function for plotting the confusion matrix for each of the Networks. 
'''


def plot_confusion(true, pred, title, filename=None):
    '''
    Creates a confusion matrix with given values
    Plotes the matrix as a heatmap

    Agruments:
        true (ndarray): The true labels
        pred (nparray): The predicted array
        title (str: Title of the plot)
    '''

    fruits = ["Apple","Banana","Kiwi","Mango","Orange","Pear","Plum"]

    cm = confusion_matrix(true, pred, normalize='true')

    sb.set(font_scale=2.0)

    heatmap = sb.heatmap(cm,cmap='YlGnBu_r',
                         xticklabels=[label for label in fruits],
                         yticklabels=[label for label in fruits],
                         cbar_kws={'label': 'Accuracy'},
                         fmt = ".2",
                         edgecolor="none",
                         annot = True,
                         annot_kws={'size':26})
    heatmap.set_xlabel('Estimated Labels', size=26)
    heatmap.set_ylabel('True Labels', size=26)
    heatmap.set_title(title, size=30)
    fig = heatmap.get_figure()
    plt.yticks(rotation=0)


    if filename:
        plt.savefig('../Figs/' + filename)


'''
__________________________________
______TESTER Neural Network_______

'''

def plot_NN():

    X_train, X_test, z_train, z_test, all_paths, labels,n_imgs = read_fruit(isNN=True, set_limit=800)

    # SETT INN BETSE PARAMETERE HER :-)
    bsize = 4
    n_epochs = 6
    eta = 0.01
    lmd = 0.0001
    n_hidden = [300, 300]


    NN_k = NN_Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, lmd, loss='categorical_crossentropy',
                    metrics=['categorical_accuracy'], act_hidden='relu', act_out='softmax')

    model = NN_k.NN_model()

    model.fit(X_train, z_train, epochs=n_epochs, batch_size=bsize, verbose=0)
    scores = model.evaluate(X_test, z_test)
    scores_train = model.evaluate(X_train, z_train)
    print(f"Accuracy test keras {scores[1]}")
    print(f"Accuracy train keras {scores_train[1]}")

    prediction = np.argmax(model.predict_step(X_test), axis=1)
    true_label = np.argmax(z_test, axis=1)

    print('pred', prediction, '\ntrue', true_label)
    print(type(prediction), type(true_label))

    plt.figure(figsize=[15,10])
    plot_confusion(true_label, prediction, 'Neural Network Prediction', filename='figs_NN/confusion_NN')


'''
__________________________________
_________TESTER CNN_______________

'''

def plot_CNN():
    X_train, X_test, z_train, z_test, _, _, _ = read_fruit(set_limit=800)


    #SETT INN BESTE PARAMETERE HER :-)
    input_shape = (75, 75, 3)
    receptive_field = 3
    n_filters = 20
    n_neurons_connected = 100
    n_categories = 7
    eta = 0.02
    lmbd = 0.0001

    b_size = 8
    epochs = 6
    act_func = 'relu'


    cnn = CNN_Keras(input_shape=input_shape,
                    receptive_field=receptive_field,
                    n_filters = n_filters,
                    n_neurons_connected = n_neurons_connected,
                    n_categories = n_categories,
                    eta = eta,
                    lmbd = lmbd,
                    act=act_func)

    model = cnn.CNN_model()

    model.fit(X_train, z_train, epochs=epochs,batch_size=b_size, verbose=1)

    train_acc = model.evaluate(X_train, z_train, verbose=1)[1]
    test_acc = model.evaluate(X_test, z_test, verbose=1)[1]
    print('_____printer accos_______')
    print(train_acc)
    print(test_acc)

    prediction = np.argmax(model.predict_step(X_test), axis=1)
    true_label = np.argmax(z_test, axis=1)

    print(prediction)
    print(true_label)

    plt.figure(figsize=[15,10])
    plot_confusion(true_label, prediction, 'CNN Prediction')
    plt.savefig('../Figs/figs_CNN/confusion_CNN')



def plot_own_NN():
    X_train, X_test, z_train, z_test, _, _, _ = read_fruit(isNN=True, set_limit=50)

    print('Testing eour own NN ...')
    bsize = 4
    epochs = range(1,11)
    eta = 0.01
    lmd = 0.0001
    n_hidden = [300,300]
    act_hidden = act.Activation_ReLU()
    act_out = act.Activation_Softmax()
    cost_func = cost.CrossEntropy()

    n_in = X_train.shape[1]
    n_out = z_train.shape[1]

    network = NeuralNetwork(n_in, n_out, act_hidden, act_out, cost_func, n_in_each_hidden=n_hidden)
    network.create_layers()

    z_tilde = network.feed_forward(X_test)


    print('Accuracy before training:', funcs.Accuracy(np.argmax(z_tilde, axis=1), np.argmax(z_test, axis=1)))

    acc_train, acc_test = train_network_class(network, bsize, len(epochs), eta, lmd, X_train, X_test, z_train, z_test)

    # Runs trained neural network with testing data
    z_tilde_train = np.argmax(network.feed_forward(X_train),axis=1)
    z_tilde_test = np.argmax(network.feed_forward(X_test), axis=1)


    print(f"Accuracy after training train: {funcs.Accuracy(z_tilde_train, np.argmax(z_train, axis=1))}")
    print(f"Accuracy after training test: {funcs.Accuracy(z_tilde_test, np.argmax(z_test, axis=1))}")


    print(acc_test[-1], acc_train[-1])

    z_test = np.argmax(z_test, axis=1)

    print(z_test, z_tilde_test)
    print(type(z_test), type(z_tilde_test))

    plt.figure(figsize=[15,10])
    plot_confusion(z_test, z_tilde_test, 'Own Network Prediction')
    plt.savefig('../Figs/figs_NN/confusion_ownNN')



#plot_own_NN()
plot_NN()
#plot_CNN()
