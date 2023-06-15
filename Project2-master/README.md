# Project2

(This readme-file is added after the deadline for Project 2)

This is the source code of Project 2 for Tiril A. Gjerstad and Helene Wold in FYS-STK4155 Autumn of 2021.

Stocastic gradient decent is implemented in one file, **SDG.py**

To implement our own Feed Forward Neural Network, we use classes. The implementation is implemented in several files, 

**NeuralNetwork.py**: The main implementation of the neural network. 

**activation.py**: Implementation of the activation functions used in the network.

**costfuncs.py**: Implementation of the cost functions used in the network. 
 
It is created *test-files* to run all the tests,

**test_NN_franke.py**: Testing neural network on franke data.

**test_SDG_franke.py**: Testing sdg on franke data.

**test_bcdata.py**: Testing neural network and logistic regression on breat cancer data.

To run the test, run:

```python
$ python3 test_NN_franke.py
$ python3 test_NN_franke.py
$ python3 test_NN_franke.py
```

You need to have installed tensorflow and sklearn for this.

**analysis_funcs.py** contains functions used in testing. 

We also used kreas to compare our results, the implementation is in the file, 

**NN_kreas.py**

**funcs.py** and **linear_regression.py** is reused from project 1. Containing implementation of some theoretical functions used in linear regression. 



