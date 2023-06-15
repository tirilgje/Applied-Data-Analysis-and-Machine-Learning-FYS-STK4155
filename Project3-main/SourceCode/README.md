# Project 3 - Code 


This is the source code of Project 3 for Tiril A. Gjerstad and Helene Wold in FYS-STK4155 Autumn of 2021.

To run the code in this project the user need to have some packages installed. These are tensorflow, sklearn and pillow. 

```python
$ pip install tensorflow
$ pip install sklearn
$ pip install pillow
```

To implement Neural Network and CNN we use Tensor flow. The algorithm is implemented in the files,

**NN_Keras.py**: The main implementation of the neural network. 

**CNN_Keras.py**: The main implementation of the CNN.


There is created own files for reading the data, 

**data_prep.py**: Reading the Fruit data from path. Prepare the data for analysis. Implemented as a class DataPrep.

**FruitReads.py**: Contains a function using the DataPrep class to read the data and returns test and train data 

 
It is created *test-files* to run all the tests,

**test_NN.py**: Testing neural network on franke data.

**test_CNN.py**: Testing CNN 

To run the tests, run:

```python
$ python3 test_NN.py
$ python3 test_CNN.py
$ python3 test_NN.py
```

**confusion.py** Plots confusion matrix for a given run of the networks. 

We reused our own Neural Network code, containing the files **NeuralNetwork.py**, **activation.py**, **costfuncs.py**. 
**analysis_funcs.py** contains functions used in testing. 

Other files used are, 

**downsizeimg.py** Downsizes and plot an image to show in the report. 

**funcs.py** is reused from project 1. Containing implementation of some theoretical functions used in linear regression. 
