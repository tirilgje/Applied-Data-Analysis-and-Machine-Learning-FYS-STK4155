import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from NN_keras import NN_Keras

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

from data_prep import DataPrep

'''
This file contains a function used to read the data we use to do our tests. 
We use this function both for NN and CNN, we can send in number of images (set_limit) if wanted. 
'''



def read_fruit(isNN=False, scale_255=False, set_limit=False):
    # Liste med alle paths
    all_paths = ["../Images/Apple/All_Apples",
                "../Images/Banana",
                "../Images/Kiwi/All_Kiwis",
                "../Images/Mango",
                "../Images/Orange",
                "../Images/Pear",
                "../Images/Plum"]

    #Alle labels
    labels = ["Apple",
            "Banana",
            "Kiwi",
            "Mango",
            "Orange",
            "Pear",
            "Plum"]



    n_fruit = len(all_paths)         # Amount of fruits categories chosen to classify

    n_imgs = []
    for i in range(len(all_paths)):
        n_imgs.append(len(os.listdir(all_paths[i])))
    print("Amount of images in each path:    ", n_imgs)

    # limit number of images
    if set_limit:
        n_imgs_choise = set_limit
    else:
        n_imgs_choise = round(min(n_imgs)/50)     # Amount of images we want to concider, 2% of smallest image amount

    print("Amount of images chosen from path:  ", n_imgs_choise)

    dp = DataPrep(path = all_paths, label = labels, n_imgs = n_imgs, limit = n_imgs_choise)

    print("Amound of images gathered:          ", np.size(dp.true_imgs_pix))
    
    dp.downsize_pixels()  

    flat = False 

    if isNN or not scale_255:
        #flatten only for neiral network (not for CNN)
        flat = True
        dp.flatten()

    images_pix = dp.true_imgs_pix               # dette blir på en måte X
    true_images_one_hot = dp.one_hot_vector     # og dette blir da z


    #lager train og test set as always
    X_train, X_test, z_train, z_test = train_test_split(images_pix, true_images_one_hot, test_size=0.2)

    if scale_255:
        # do not want to scale z because we want to keep it a one hot vector
        X_train, X_test = X_train/255, X_test/255
    else:

        if not isNN and not flat:
            flat = True
            dp.flatten()

        # Scale the data
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        if not isNN and flat:
            #  unflatten to use CNN
            X_train = X_train.reshape(X_train.shape[0], dp.shape1, dp.shape2, dp.shape3)
            X_test = X_test.reshape(X_test.shape[0], dp.shape1, dp.shape2, dp.shape3)


            flat = False


    return X_train, X_test, z_train, z_test, all_paths, labels, n_imgs
