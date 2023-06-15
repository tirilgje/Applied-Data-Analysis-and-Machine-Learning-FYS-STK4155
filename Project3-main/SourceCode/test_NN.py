import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import activation as act
import costfuncs as cost
from NeuralNetwork import NeuralNetwork
import funcs

from FruitReads import read_fruit
from NN_keras import NN_Keras


'''
This file contains all the tests perfomed on the neural network in addition to two functions for plotting the results. 
It is 6 different tests:
    test_eta_epochs
    test_eta_lmd
    test_bsize_epochs
    test_actfunc_epochs
    test_hidden_epochs
    test_number_of_images

It is also a function to run each of these tests.
'''


def plot_heatmap(ACC, xval, yval, xlab, ylab, title, filename=None):
    '''
    Plots heatmap
    '''
    sb.set(font_scale=2.0)

    heatmap = sb.heatmap(ACC, annot=True, fmt='.2g', cmap='YlGnBu',
                     xticklabels=xval, yticklabels=yval,
                     cbar_kws={'label': 'Accuracy'}, annot_kws={'size':26})
    heatmap.set_ylabel(ylab, size=26)
    heatmap.set_xlabel(xlab, size=26)
    heatmap.invert_yaxis()
    heatmap.set_title(title, size=30)
    if filename:
        print('saving fig ', filename)
        plt.savefig('../Figs/' + filename, dpi=300)

def plot_graph(vals, epochs, label, xlab, ylab, title, ylim=[0,1.01], c=None, ls='-', filename=None):
    '''
    Plots stuff as a graph
    '''
    if c:
        plt.plot(epochs, vals, c, label = label, ls=ls)
    else:
        plt.plot(epochs, vals, label = label, ls=ls)

    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    plt.title(title, fontsize=24)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(ylim)
    plt.legend(prop={'size':16})
    if filename:
        plt.savefig('../Figs/' + filename, dpi=300)

#Test 1: Tilpass eta + epochs
def test_eta_epochs(epochs, etas):
    '''
    Testing different eta vs epochs for NN
    Calculates the accuracy score for both test and train data.

    Arguements:
        epochs (list(int)): list of epochs
        etas (list(int)): List og etas

    Returns:
        ACCs_test (list(float)): List of accuracy score for the test data
        ACCs_train (list(float)): List of accuracy score for the train data
    '''
    # Setter paramterere for en enkel model:

    act = 'relu'       # relu siden denne er enkel (og vi vet den er ganske decent)
    n_hidden = [100]   # starter enkelt med 1 hidden layer
    bsize = 4          # 4 er lavt tall gir lavere computational cost
    lmd = 0            # starter enkelt med lmd = 0


    ACCs_test = np.zeros((len(etas), len(epochs)))
    ACCs_train = np.zeros((len(etas), len(epochs)))


    print('\nTesting eta vs epohcs for Neural Network ...')
    for i in range(len(etas)):
        print('\n------', i+1, '/', len(etas), '-', etas[i], '------')
        for j in range(len(epochs)):
            nn_k = NN_Keras(n_hidden, X_train.shape[1], z_train.shape[1], etas[i], lmd, loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'], act_hidden=act, act_out='softmax')

            model = nn_k.NN_model()

            model.fit(X_train, z_train, epochs=epochs[j], batch_size=bsize, verbose=0)
            scores_test = model.evaluate(X_test, z_test)
            scores_train = model.evaluate(X_train, z_train)
            print(f"Accuracy test  keras {scores_test[1]}")
            print(f"Accuracy train keras {scores_train[1]}")

            ACCs_test[i,j] = scores_test[1]
            ACCs_train[i,j] = scores_train[1]

    return ACCs_test, ACCs_train

# Test 2: Tilpass eta + lambda
def test_eta_lmd(lmds, etas):
    '''
    Testing different lambdas and etas for NN
    Calculates the accuracy score for both test and train data.

    Arguements:
        lmds (list(int)): list of lambdas
        etas (list(int)): List of etas.

    Returns:
        ACCs_test (list(float)): List of accuracy score for the test data
        ACCs_train (list(float)): List of accuracy score for the train data
    '''

    # Setter parametere for enkel modell
    act = 'relu'
    n_hidden = [100]
    bsize = 4

    epochs = 6  # train stabiliserer seg rundt epoch 4, safe å velge 6


    ACCs_test = np.zeros((len(etas), len(lmds)))
    ACCs_train = np.zeros((len(etas), len(lmds)))


    print('\nTesting eta vs lmd for Neural Network ....')
    for i in range(len(etas)):
        print('\n------', i+1, '/', len(etas), '-', etas[i], '------')

        for j in range(len(lmds)):
            nn_k = NN_Keras(n_hidden, X_train.shape[1], z_train.shape[1], etas[i], lmds[j], loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'], act_hidden=act, act_out='softmax')

            model = nn_k.NN_model()

            model.fit(X_train, z_train, epochs=epochs, batch_size=bsize, verbose=0)
            scores_test = model.evaluate(X_test, z_test)
            scores_train = model.evaluate(X_train, z_train)
            print(f"Accuracy test keras {scores_test[1]}")
            print(f"Accuracy train keras {scores_train[1]}")

            #prediction = np.argmax(model.predict_step(X_test), axis=1)
            #true_label = np.argmax(z_test, axis=1)

            ACCs_test[i,j] = scores_test[1]
            ACCs_train[i,j] = scores_train[1]

    return ACCs_test, ACCs_train

#Test 3: Tilpass batch size + epochs
def test_bsize_epochs(epochs, bsizes):
    '''
    Testing different eta vs epochs for NN
    Calculates the accuracy score for both test and train data.

    Arguements:
        epochs (list(int)): list of epochs
        bsizes (list(int)): List of batch sizes

    Returns:
        ACCs_test (list(float)): List of accuracy score for the test data
        ACCs_train (list(float)): List of accuracy score for the train data
    '''
    # Setter paramterere for en enkel model:
    act = 'relu'
    n_hidden = [100]

    # Fra tidligere tester
    eta = 0.01
    lmd = 0.0001


    ACCs_test = np.zeros((len(bsizes), len(epochs)))
    ACCs_train = np.zeros((len(bsizes), len(epochs)))


    print('\nTesting batch size vs epohcs for Neural Network ...')
    for i in range(len(bsizes)):
        print('\n------', i+1, '/', len(bsizes), '-', bsizes[i], '------')
        for j in range(len(epochs)):
            nn_k = NN_Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, lmd, loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'], act_hidden=act, act_out='softmax')

            model = nn_k.NN_model()

            model.fit(X_train, z_train, epochs=epochs[j], batch_size=bsizes[i], verbose=0)
            scores_test = model.evaluate(X_test, z_test)
            scores_train = model.evaluate(X_train, z_train)
            print(f"Accuracy test  keras {scores_test[1]}")
            print(f"Accuracy train keras {scores_train[1]}")

            ACCs_test[i,j] = scores_test[1]
            ACCs_train[i,j] = scores_train[1]

    return ACCs_test, ACCs_train

# Test 4: Tilpass activation function + epochs
def test_actfunc_epochs(actfuncs, epochs):
    '''
    Testing different activation functions NN
    Calculates the accuracy score for both test and train data.

    Arguements:
        epochs (list(int)): List with number of epochs
        actfuncs (list(str)): List of activationfunctions

    Returns:
        ACCs_test (list(float)): List of accuracy score for the test data
        ACCs_train (list(float)): List of accuracy score for the train data
    '''

    n_hidden = [100]
    bsize = 4
    lmd = 0.0001
    eta = 0.01


    ACCs_test = np.zeros((len(actfuncs), len(epochs)))
    ACCs_train = np.zeros((len(actfuncs), len(epochs)))


    print('\nTesting activation functions for Neural Network ....')
    for i in range(len(actfuncs)):
        print('\n------', i+1, '/', len(actfuncs), '-', actfuncs[i], '------')
        for j in range(len(epochs)):
            nn_k = NN_Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, lmd, loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'], act_hidden=actfuncs[i], act_out='softmax')

            model = nn_k.NN_model()

            model.fit(X_train, z_train, epochs=epochs[j], batch_size=bsize, verbose=0)
            scores_test = model.evaluate(X_test, z_test)
            scores_train = model.evaluate(X_train, z_train)
            print(f"Accuracy test  keras {scores_test[1]}")
            print(f"Accuracy train keras {scores_train[1]}")

            ACCs_test[i,j] = scores_test[1]
            ACCs_train[i,j] = scores_train[1]

    return ACCs_test, ACCs_train

# Test 5: Tilpass hidden layers + epochs
def test_hidden_epochs(n_hiddens, epochs):
    '''
    Testing different hidden layers and batch sizes for NN
    Calculates the accuracy score for both test and train data.

    Arguements:
        n_hiddens (list(list(int))): list of lists containing number of nodes in each hidden layer
        bsizes (list(int)): List containg bsizes.

    Returns:
        ACCs_test (list(float)): List of accuracy score for the test data
        ACCs_train (list(float)): List of accuracy score for the train data
    '''


    act = 'relu'
    eta = 0.01
    lmd = 0.0001
    bsize = 4

    ACCs_test = np.zeros((len(n_hiddens), len(epochs)))
    ACCs_train  = np.zeros((len(n_hiddens), len(epochs)))


    print('\nTesting hidden layers vs epochs for Neural Network ....')
    for i in range(len(n_hiddens)):
        print('\n------', i+1, '/', len(n_hiddens), '-', n_hiddens[i], '------')
        for j in range(len(epochs)):

            nn_k = NN_Keras(n_hiddens[i], X_train.shape[1], z_train.shape[1], eta, lmd, loss='categorical_crossentropy',
                            metrics=['categorical_accuracy'], act_hidden=act, act_out='softmax')

            model = nn_k.NN_model()

            model.fit(X_train, z_train, epochs=epochs[j], batch_size=bsize, verbose=0)
            scores_test = model.evaluate(X_test, z_test)
            scores_train = model.evaluate(X_train, z_train)
            print(f"Accuracy test keras {scores_test[1]}")
            print(f"Accuracy train keras {scores_train[1]}")

            ACCs_test[i,j] = scores_test[1]
            ACCs_train[i,j] = scores_train[1]

    return ACCs_test, ACCs_train


# Test 6: Tilpass antall bilder
def test_number_of_images(n_imgs):

    eta = 0.01
    lmd = 0.0001
    bsize = 4
    epochs = 6
    act = 'relu'
    n_hidden = [300, 300]

    ACCs_test = np.zeros((len(n_imgs)))
    ACCs_train = np.zeros((len(n_imgs)))


    print('\nTesting number of images for Neural Network ...')
    for i in range(len(n_imgs)):
        X_train, X_test, z_train, z_test, _, _, _ = read_fruit(isNN=True, set_limit=n_imgs[i])

        print('\n------', i+1, '/', len(n_imgs), '-', n_imgs[i], '------')

        nn_k = NN_Keras(n_hidden, X_train.shape[1], z_train.shape[1], eta, lmd, loss='categorical_crossentropy',
                        metrics=['categorical_accuracy'], act_hidden=act, act_out='softmax')

        model = nn_k.NN_model()

        model.fit(X_train, z_train, epochs=epochs, batch_size=bsize, verbose=1)
        scores_test = model.evaluate(X_test, z_test)
        scores_train = model.evaluate(X_train, z_train)
        print(f"Accuracy test  keras {scores_test[1]}")
        print(f"Accuracy train keras {scores_train[1]}")

        ACCs_test[i] = scores_test[1]
        ACCs_train[i] = scores_train[1]

    return ACCs_test, ACCs_train




'''
_______________________________
______TESTING STARTER HER______

'''

#Test 1 - ETA vs EPOCHS
def run_test1():
    plt.figure(figsize = [15, 10])

    # FINN UT HVILKE VERIDER VI VIL TESTE MED
    etas = [1e-4, 1e-3, 5e-3, 1e-2, 2e-2]
    epochs = range(1,11)

    accs_ep_test, accs_ep_train = test_eta_epochs(epochs, etas)
    colors = ['tomato', 'limegreen', 'deepskyblue', 'violet', 'slateblue']


    for i in range(len(etas)):
        plot_graph(accs_ep_test[i], epochs, etas[i],'Epochs', 'Accuracy', 'Etas (NN)', c=colors[i])
        plot_graph(accs_ep_train[i], epochs, str(etas[i]) + ' train', 'Epochs', 'Accuracy', 'Etas (NN)', c=colors[i], ls=':')

    plt.savefig('../Figs/figs_NN/eta_ep200')
    #plt.show()

# Test 2 - ETA vs LAMBDA
def run_test2():
    plt.figure(figsize = [15, 10])

    # Tester forskjellige
    lmds = [0, 1e-4, 1e-2, 0.05, 0.1, 0.15, 0.2, 0.3]

    # Test verdier mellom 0.001 og 0.01
    etas = [0.001, 0.003, 0.005, 0.008, 0.01]

    ACCs_test, ACCs_train = test_eta_lmd(lmds, etas)

    plot_heatmap(ACCs_test, lmds, etas,
                'Lambda', 'Eta',
                'Eta vs Lambda (NN)', filename='figs_NN/eta_lmd200')
    #plt.show()

# Test 3 - BATCH SIZE vs EPOCHS
def run_test3():
    plt.figure(figsize = [15, 10])

    # FINN UT HVILKE VERIDER VI VIL TESTE MED
    bsizes = [2, 4, 8, 16, 32]
    epochs = range(1,11)

    accs_ep_test, accs_ep_train = test_bsize_epochs(epochs, bsizes)
    colors = ['tomato', 'limegreen', 'deepskyblue', 'violet', 'slateblue', 'slategray']


    for i in range(len(bsizes)):
        plot_graph(accs_ep_test[i], epochs, bsizes[i],'Epochs', 'Accuracy', 'Batch size (NN)', c=colors[i])
        plot_graph(accs_ep_train[i], epochs, str(bsizes[i]) + ' train', 'Epochs', 'Accuracy', 'Batch size (NN)', c=colors[i], ls=':')

    plt.savefig('../Figs/figs_NN/bsize_ep200')
    #plt.show()

# Test 4 - ACTIVATION FUNCTIONS vs EPOCHS
def run_test4():

    plt.figure(figsize = [15, 10])

    act_funcs = ['relu', 'elu', 'sigmoid', 'tanh']
    epochs = range(1,11)

    accs_ep_test, accs_ep_train = test_actfunc_epochs(act_funcs, epochs)
    colors = ['tomato', 'limegreen', 'deepskyblue', 'darkviolet']


    for i in range(len(act_funcs)):
        plot_graph(accs_ep_test[i], epochs, act_funcs[i],'Epochs', 'Accuracy', 'Activation functions (NN)', c=colors[i])
        plot_graph(accs_ep_train[i], epochs, str(act_funcs[i]) + ' train', 'Epochs', 'Accuracy', 'Activation functions (NN)', c=colors[i], ls=':')

    plt.savefig('../Figs/figs_NN/actfuncs200')
    #plt.show()

# Test 5 - HIDDEN LAYERS vs EPOCHS
def run_test5():

    plt.figure(figsize = [24, 14])

    n_hiddens = [[100], [200], [500], [300,300], [500,500], [100,100,100],[500,500,500], [300, 100, 50, 10], [10, 50, 100, 300]]

    epochs = range(1,7)

    ACCs_test, ACCs_train = test_hidden_epochs(n_hiddens, epochs)

    plot_heatmap(ACCs_test, epochs, n_hiddens,
                'Epochs', 'Hidden layers',
                'Hidden layers vs epochs (NN)', filename='figs_NN/hidden200')

    #plt.show()

    plt.figure(figsize=[15,10])
    colors = ['tomato', 'limegreen', 'deepskyblue', 'violet', 'slateblue', 'slategray', 'olive', 'peru', 'lightpink']


    for i in range(len(n_hiddens)):
        plot_graph(ACCs_test[i], epochs, n_hiddens[i],'Epochs', 'Accuracy', 'Hidden layers (NN)', c=colors[i])
        plot_graph(ACCs_train[i], epochs, str(n_hiddens[i]) + ' train', 'Epochs', 'Accuracy', 'Hidden layers (NN)', c=colors[i], ls=':')

    plt.savefig('../Figs/figs_NN/hidden_ep200')



# Test 6: NUMBER OF IMAGES
def run_test6():

    plt.figure(figsize = [15, 10])

    n_imgs = [100, 200, 300, 400, 600, 800, 1000]
    ACCs_test, ACCs_train = test_number_of_images(n_imgs)

    print(ACCs_test)
    print(ACCs_train)



    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(111)
    plt.plot(n_imgs, ACCs_test, '-o', label = 'Test', linewidth=3.0)
    plt.plot(n_imgs, ACCs_train, label = 'Train', ls=':', linewidth=3.0)
    plt.title('Testing number of images (NN)')
    plt.xlabel('Number of images')
    plt.ylabel('Accuracy')
    plt.ylim([0.6,1.01])
    plt.legend()

    for x,y in zip(n_imgs,ACCs_test):

        label = "{:.2f}".format(y)

        plt.annotate(label, # this is the text
                    (x,y), # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,15), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center


    plt.savefig('../Figs/figs_NN/n_imgs')


'''
______________________________
______Vårt eget NN fra P2_____

'''

def train_network_class(network, batch_size, n_epochs, eta, lmd, X_train, X_test, z_train, z_test):
    '''
    Takes a network and parameter inputs as input and train the nettwork as a classification problem
    Can be applied to both neural network and logisitc regression classification.
    Returns the accuracy scores for both test and train.
    '''

    minibatches = int(z_train.shape[0]/batch_size)

    # train the network using minibatches
    training_accs = []
    test_accs = []

    index = np.arange(len(X_train))
    for k in range(n_epochs):
        print('Epoch:', k+1, '/', n_epochs)
        np.random.shuffle(index)
        X_batches = np.array_split(X_train[index], minibatches)
        z_batches = np.array_split(z_train[index], minibatches)

        for l in range(minibatches):
            r = np.random.randint(0, minibatches-1)
            network.backprop(X_batches[r], z_batches[r], eta=eta, lmd=lmd)

        z_tilde_train = np.argmax(network.feed_forward(X_train),axis=1)
        z_tilde_test = np.argmax(network.feed_forward(X_test), axis=1)


        acc_train = funcs.Accuracy(z_tilde_train, np.argmax(z_train, axis=1))
        acc_test = funcs.Accuracy(z_tilde_test, np.argmax(z_test, axis=1))

        training_accs.append(acc_train)
        test_accs.append(acc_test)

    return training_accs, test_accs

def test_own_NN():
    print('Testing eour own NN ...')
    bsize = 8
    epochs = range(1,21)
    eta = 0.1
    lmd = 0.0001
    n_hidden = [300]
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
    plot_graph(acc_test, epochs, 'Test', 'Epochs', 'Accuracy', 'Own implemented NN', ylim=[0,1.01])
    plot_graph(acc_train, epochs, 'Train', 'Epochs', 'Accuracy', 'Own implemented NN', ylim=[0,1.01])
    plt.show()



'''
_________________________________
_____VELG TESTER Å KJØRE HER_____

'''

if __name__ == '__main__':
    #Alle testene bruker samme data som er tilgjengelig i hele skriptet
    X_train, X_test, z_train, z_test, _, _,_ = read_fruit(isNN=True, set_limit=200)

    run_test1()  # eta vs epochs
    run_test2()  # eta vs lambda
    run_test3()  # batsh size vs epochs
    run_test4()  # activation functions vs epochs
    run_test5()  # hidden layers vs epochs
    run_test6()   # number of images
    test_own_NN()  # test our own Neural network
