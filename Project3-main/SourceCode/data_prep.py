from PIL import Image
import numpy as np
import os
import random 


'''
This fine contains the DataPrep class 
Here the paths to the images is sent together with the labels, to prepare the data for the analysis. 
The data is read, downsamples and flattened is needed. 
'''


class DataPrep:
    def __init__(self, path, label, n_imgs, limit = None):
        """
        Reading the images 

        Arguments: 
            path (list(str)): Path to the images in each cateory
            label (list(str)): The name of each category
            n_imgs (list(int)): Number of element in each category
            limit (int) [optional, default=None]: Number of images to use
        """

        random.seed(1997)

        self.path = path     # 7 paths 
        self.label = label   # 7 labels 
        self.n_imgs = n_imgs # 7 ints 
        if limit:
            # Limit er hvis vi ikke vil bruke ALLE bildene i hver kategori
            self.limit = limit # 1 int 

        # Reading starts 
        self.true_imgs_pix = []    # (7 x limit antall bilder)
        self.true_lbl = []         #  (7 x limit antall lbls)

        # Looping through the caterories 
        for i in range(len(path)):
            
            if limit:
                # choosing random images 
                choosen_sample = random.sample(range(n_imgs[i]), limit) 
            else: 
                # alle bilder i alle kategirer er med 
                choosen_sample = range(n_imgs[i])

            for j in choosen_sample:
                self.true_lbl.append(label[i])            # legger inn true label i lbl
                # Leser bildet
                p = path[i]
                name = os.listdir(p)[j]
                path_to_image = os.path.join(p, name)            # Henter bildenavnet
                image = Image.open(path_to_image)                # Leser selve bildet
                self.true_imgs_pix.append(np.array(image))       # Lagrer bildet

        self.true_lbl = np.array(self.true_lbl)
        self.true_imgs_pix = np.array(self.true_imgs_pix)

        # one hot er true imahes i one hot form aka det vi vil softmax skal spytte ut 
        self.one_hot_vector = np.zeros((len(self.true_imgs_pix), len(label)))
        eye = np.eye(len(label))
        for i in range(len(label)):
            self.one_hot_vector[np.where(self.true_lbl == label[i])] = eye[i]


    def downsize_pixels(self, n_pixels = 75):
        '''
        Arguments:
            n_pixels (int) [optional, default=100]: Number of pixels after downsapling
        '''
        # Create the array for new imgs
        new_imgs = np.zeros((int(self.true_imgs_pix.shape[0]), n_pixels, n_pixels,
                             int(self.true_imgs_pix[0].shape[-1])))
        
        # Henter ut pixels fra orginal img
        for i in range(len(self.true_imgs_pix)):
            x,y,_ = self.true_imgs_pix[i].shape
            skip_x = np.linspace(0,x-1,n_pixels).astype("int")
            skip_y = np.linspace(0,y-1,n_pixels).astype("int")
            im = self.true_imgs_pix[i][skip_x,:]
            im = im[:,skip_y]
            new_imgs[i] = im

        self.true_imgs_pix = np.zeros((new_imgs.shape))
        self.true_imgs_pix[:] = new_imgs


    def flatten(self):
        '''
        Flatten img from 4d to 2d 
        Used in NN 
        '''

        print("Shape before:", self.true_imgs_pix.shape[0],
                               self.true_imgs_pix.shape[1], 
                               self.true_imgs_pix.shape[2], 
                               self.true_imgs_pix.shape[3])
        #save shapes
        self.shape1 = self.true_imgs_pix.shape[1]
        self.shape2 = self.true_imgs_pix.shape[2]
        self.shape3 = self.true_imgs_pix.shape[3]


        self.true_imgs_pix = self.true_imgs_pix.reshape(self.true_imgs_pix.shape[0],
                                                self.true_imgs_pix.shape[1]*self.true_imgs_pix.shape[2]*self.true_imgs_pix.shape[3])

        print("Shape after:", self.true_imgs_pix.shape[0], 
                              self.true_imgs_pix.shape[1])


