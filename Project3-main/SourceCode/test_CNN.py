import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from FruitReads import read_fruit
from CNN_keras import CNN_Keras

'''
This file contains all the tests perfomed on the neural network in addition to two functions for plotting the results. 
It is 7 different tests:
    test_eta_epochs
    test_eta_lmd
    test_bsize_epochs
    test_actfunc_epochs
    test_recfields_nfilts
    test_nneurons_epochs
    test_number of images

It is also a function to run each of these tests.
'''

#Alle testene bruker samme data som er tilgjengelig i hele skriptet 
X_train, X_test, z_train, z_test, _, labels, _ = read_fruit(set_limit=200)

input_shape = (75,75,3)
n_cat = len(labels)


def plot_heatmap(ACC, xval, yval, xlab, ylab, title, filename=None):
    '''
    Plots heatmap 
    '''

    sb.set(font_scale=2.0)

    heatmap = sb.heatmap(ACC, annot=True, fmt='.2g', cmap='YlGnBu',
                     xticklabels=xval, yticklabels=yval,
                     cbar_kws={'label': 'Accuracy'}, annot_kws={"size":26})
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

    rec_field = 3
    n_filt = 20
    n_neurons = 100
    lmd = 0
    act = 'relu'
    bsize = 4


    ACCs_test = np.zeros((len(etas), len(epochs)))
    ACCs_train = np.zeros((len(etas), len(epochs)))


    print('\nTesting eta vs epohcs for CNN ...')
    for i in range(len(etas)):
        print('\n------', i+1, '/', len(etas), '-', etas[i], '------')
        for j in range(len(epochs)):
            
            cnn_k = CNN_Keras(input_shape, rec_field, n_filt, n_neurons, n_cat, etas[i], lmd, act)


            model = cnn_k.CNN_model()

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
    
    # Setter paramterere for en enkel model: 

    rec_field = 3
    n_filt = 20
    n_neurons = 100
    act = 'relu'
    bsize = 4
    epochs = 6


    ACCs_test = np.zeros((len(etas), len(lmds)))
    ACCs_train = np.zeros((len(etas), len(lmds)))


    print('\nTesting eta vs lmd for CNN ....')
    for i in range(len(etas)):
        print('\n------', i+1, '/', len(etas), '-', etas[i], '------')

        for j in range(len(lmds)):
            
            cnn_k = CNN_Keras(input_shape, rec_field, n_filt, n_neurons, n_cat, etas[i], lmds[j], act)

            model = cnn_k.CNN_model()

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
    rec_field = 3
    n_filt = 20
    n_neurons = 100
    act = 'relu'

    # Fra tidligere tester 
    eta = 0.02
    lmd = 0.0001            


    ACCs_test = np.zeros((len(bsizes), len(epochs)))
    ACCs_train = np.zeros((len(bsizes), len(epochs)))


    print('\nTesting batch size vs epohcs for CNN ...')
    for i in range(len(bsizes)):
        print('\n------', i+1, '/', len(bsizes), '-', bsizes[i], '------')
        for j in range(len(epochs)):
            cnn_k = CNN_Keras(input_shape, rec_field, n_filt, n_neurons, n_cat, eta, lmd, act)

            model = cnn_k.CNN_model()

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
    
    rec_field = 3
    n_filt = 20
    n_neurons = 100
    lmd = 0.0001
    bsize = 8
    eta = 0.02

    ACCs_test = np.zeros((len(actfuncs), len(epochs)))
    ACCs_train = np.zeros((len(actfuncs), len(epochs)))


    print('\nTesting activation functions for CNN ....')
    for i in range(len(actfuncs)):
        print('\n------', i+1, '/', len(actfuncs), '-', actfuncs[i], '------')
        for j in range(len(epochs)):
            cnn_k = CNN_Keras(input_shape, rec_field, n_filt, n_neurons, n_cat, eta, lmd, actfuncs[i])


            model = cnn_k.CNN_model()

            model.fit(X_train, z_train, epochs=epochs[j], batch_size=bsize, verbose=0)
            scores_test = model.evaluate(X_test, z_test)
            scores_train = model.evaluate(X_train, z_train)
            print(f"Accuracy test  keras {scores_test[1]}")
            print(f"Accuracy train keras {scores_train[1]}")

            ACCs_test[i,j] = scores_test[1]
            ACCs_train[i,j] = scores_train[1]
    
    return ACCs_test, ACCs_train

# Test 5: Tilpass rec fields + n filters  
def test_recfield_nfilts(recfields, nfilts):
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

    n_neurons = 100
    lmd = 0.0001
    act = 'relu'
    bsize = 8 
    eta = 0.02
    epochs = 8   

    ACCs_test = np.zeros((len(recfields), len(nfilts)))
    ACCs_train  = np.zeros((len(recfields), len(nfilts)))
    

    print('\nTesting hidden layers vs epochs for CNN ....')
    for i in range(len(recfields)):
        print('\n------', i+1, '/', len(recfields), '-', recfields[i], '------')
        for j in range(len(nfilts)):

            cnn_k = CNN_Keras(input_shape, recfields[i], nfilts[j], n_neurons, n_cat, eta, lmd, act)


            model = cnn_k.CNN_model()

            model.fit(X_train, z_train, epochs=epochs, batch_size=bsize, verbose=0)
            scores_test = model.evaluate(X_test, z_test)
            scores_train = model.evaluate(X_train, z_train)
            print(f"Accuracy test keras {scores_test[1]}")
            print(f"Accuracy train keras {scores_train[1]}")

            ACCs_test[i,j] = scores_test[1]
            ACCs_train[i,j] = scores_train[1]

    return ACCs_test, ACCs_train

# Test 6: Tilpass number of neurons connected 
def test_nneurons_epochs(n_neurons, epochs):
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

    n_filt = 20
    lmd = 0.0001
    act = 'relu'
    bsize = 8 
    eta = 0.02 
    recfield = 3  

    ACCs_test = np.zeros((len(n_neurons), len(epochs)))
    ACCs_train  = np.zeros((len(n_neurons), len(epochs)))
    

    print('\nTesting number of neurons connected vs epochs for CNN ....')
    for i in range(len(n_neurons)):
        print('\n------', i+1, '/', len(n_neurons), '-', n_neurons[i], '------')
        for j in range(len(epochs)):

            cnn_k = CNN_Keras(input_shape, recfield, n_filt, n_neurons[i], n_cat, eta, lmd, act)


            model = cnn_k.CNN_model()

            model.fit(X_train, z_train, epochs=epochs[j], batch_size=bsize, verbose=0)
            scores_test = model.evaluate(X_test, z_test)
            scores_train = model.evaluate(X_train, z_train)
            print(f"Accuracy test keras {scores_test[1]}")
            print(f"Accuracy train keras {scores_train[1]}")

            ACCs_test[i,j] = scores_test[1]
            ACCs_train[i,j] = scores_train[1]

    return ACCs_test, ACCs_train


# Test 7: Tilpass antall bilder 
def test_number_of_images(n_imgs):

    rec_field = 3
    n_filt = 20
    n_neurons = 100
    lmd = 0.0001
    act = 'relu'
    bsize = 4
    eta = 0.01
    epochs = 6

    ACCs_test = np.zeros((len(n_imgs)))
    ACCs_train = np.zeros((len(n_imgs)))


    print('\nTesting number of images for CNN ...')
    for i in range(len(n_imgs)):
        X_train, X_test, z_train, z_test, _, _, _ = read_fruit(set_limit=n_imgs[i])

        print('\n------', i+1, '/', len(n_imgs), '-', n_imgs[i], '------')

        cnn_k = CNN_Keras(input_shape, rec_field, n_filt, n_neurons, n_cat, eta, lmd, act)

        model = cnn_k.CNN_model()

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
    etas = [1e-4, 1e-3, 1e-2, 2e-2, 3e-2]
    epochs = range(1,9)

    accs_ep_test, accs_ep_train = test_eta_epochs(epochs, etas)
    colors = ['tomato', 'limegreen', 'deepskyblue', 'violet', 'slateblue']


    for i in range(len(etas)):
        plot_graph(accs_ep_test[i], epochs, etas[i],'Epochs', 'Accuracy', 'Etas (CNN)', c=colors[i])
        plot_graph(accs_ep_train[i], epochs, str(etas[i]) + ' train', 'Epochs', 'Accuracy', 'Etas (CNN)', c=colors[i], ls=':')
    
    plt.savefig('../Figs/figs_CNN/eta_ep200')
    #plt.show()

# Test 2 - ETA vs LAMBDA 
def run_test2():
    plt.figure(figsize = [15, 10])

    # Tester forskjellige 
    lmds = [0, 1e-4, 1e-3, 1e-2, 0.1, 0.2]

    # Test verdier mellom 0.001 og 0.01
    etas = [0.005, 0.01, 0.015, 0.02]

    ACCs_test, ACCs_train = test_eta_lmd(lmds, etas)
    
    plot_heatmap(ACCs_test, lmds, etas, 
                'Lambda', 'Eta', 
                'Accuracy of Eta vs Lambda (CNN)', filename='figs_CNN/eta_lmd200')
    #plt.show()

# Test 3 - BATCH SIZE vs EPOCHS 
def run_test3():
    plt.figure(figsize = [15, 10])

    # FINN UT HVILKE VERIDER VI VIL TESTE MED 
    bsizes = [2, 4, 8, 16, 32]
    epochs = range(1,11,2)

    accs_test, accs_train = test_bsize_epochs(epochs, bsizes)
    colors = ['tomato', 'limegreen', 'deepskyblue', 'violet', 'slateblue', 'slategray']


    for i in range(len(bsizes)):
        plot_graph(accs_test[i], epochs, bsizes[i],'Epochs', 'Accuracy', 'Batch size (NN)', c=colors[i])
        plot_graph(accs_train[i], epochs, str(bsizes[i]) + ' train', 'Epochs', 'Accuracy', 'Batch size (NN)', c=colors[i], ls=':')
    
    plt.savefig('../Figs/figs_CNN/bsize_ep200')
    #plt.show()

    plt.figure(figsize=[15,10])

    plot_heatmap(accs_test, epochs, bsizes, 
            'Epochs', 'Batch size', 
            'Accuracy of batch size vs epochs (CNN)', filename='figs_CNN/bsize200')
    #plt.show()

# Test 4 - ACTIVATION FUNCTIONS vs EPOCHS 
def run_test4():
    
    plt.figure(figsize = [15, 10])

    act_funcs = ['relu', 'elu', 'sigmoid', 'tanh']
    epochs = range(1,7)

    accs_ep_test, accs_ep_train = test_actfunc_epochs(act_funcs, epochs)
    colors = ['tomato', 'limegreen', 'deepskyblue', 'darkviolet']


    for i in range(len(act_funcs)):
        plot_graph(accs_ep_test[i], epochs, act_funcs[i],'Epochs', 'Accuracy', 'Activation functions (CNN)', c=colors[i])
        plot_graph(accs_ep_train[i], epochs, str(act_funcs[i]) + ' train', 'Epochs', 'Accuracy', 'Activation functions (CNN)', c=colors[i], ls=':')
    
    plt.savefig('../Figs/figs_CNN/actfuncs200')
    #plt.show()

# Test 5 - REC FIELDS vs N FILTERS  
def run_test5():

    plt.figure(figsize = [18, 10])

    recfields = [3,5,7,9]
    n_filters = [10,20,30,40]



    ACCs_test, ACCs_train = test_recfield_nfilts(recfields, n_filters)


    plot_heatmap(ACCs_test, recfields, n_filters, 
                'Receptive field', 'Number of filters', 
                'Receptive fields vs number of filters (CNN)', filename='figs_CNN/recfi_filt200')

    #plt.show()


# Test 6: NUMBER OF NEURONS CONNECTED 
def run_test6():

    plt.figure(figsize = [18, 10])

    n_neurons = [50, 100, 150, 200, 300, 500]
    epochs = range(1,7)



    ACCs_test, ACCs_train = test_nneurons_epochs(n_neurons, epochs)


    plot_heatmap(ACCs_test, epochs, n_neurons, 
                'Epochs', 'Number of neurons connected',
                'Number of neurons connected (CNN)', filename='figs_CNN/n_neurons200')

    #plt.show()


# Test 6: NUMBER OF IMAGES 
def run_test7():

    plt.figure(figsize = [15, 10])

    n_imgs = [100, 200, 300, 400, 600, 800, 1000]
    ACCs_test, ACCs_train = test_number_of_images(n_imgs)
    
    # bare lagra verdiene etter en kjøring for å fikse på plottet og sleppe å kjøre på nyttt hehe 
    #ACCs_test = [0.79285717, 0.86785716, 0.90476191, 0.88928574, 0.90833336, 0.94017857, 0.93357146]
    #ACCs_train = [1., 1., 1., 1., 1., 1., 1.]

    print(ACCs_test)
    print(ACCs_train)

    fig = plt.figure(figsize=[15,10])
    ax = fig.add_subplot(111)
    plt.plot(n_imgs, ACCs_test, '-o', label = 'Test', linewidth=3.0)
    plt.plot(n_imgs, ACCs_train, label = 'Train', ls=':', linewidth=3.0)
    plt.title('Testing number of images (CNN)', fontsize=24)
    plt.xlabel('Number of images', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.ylim([0.6,1.01])
    plt.legend(prop={'size':20})

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    for x,y in zip(n_imgs,ACCs_test):

        label = "{:.2f}".format(y)

        plt.annotate(label,                     # this is the text
                    (x,y),                      # these are the coordinates to position the label
                    textcoords="offset points", # how to position the text
                    xytext=(0,15),              # distance from text to points (x,y)
                    ha='center',                # horizontal alignment can be left, right or center
                    fontsize=16) 
    
    
    plt.savefig('../Figs/figs_CNN/n_imgs', dpi=300)
    


'''
_________________________________
_____VELG TESTER Å KJØRE HER_____

'''
if __name__ == '__main__':
    run_test1()   # eta vs epochs 
    run_test2()   # eta vs lambda 
    run_test3()   # batsh size vs epochs 
    run_test4()   # activation functions vs epochs 
    run_test5()    # receptive field vs n filters 
    run_test6()   # number of neurons connected 
    run_test7()   # number of images 





