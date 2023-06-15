# Project1

This is the source code of Project 1 for Tiril A. Gjerstad and Helene Wold in FYS-STK4155 Autumn of 2021.

All data analysis, both Franke Function and real terrain data, is implemented in ols.py, ridge.py and lasso.py - but also possible to implement through the bootstrap.py and cross_validation.py files.

## funcs.py

This file contains several implementations of theoretical equations and different methods used throughout the entire project. All functions are called using func.FUNCTIONNAME throughout the project.

## linear_regression.py

Contains the main linear regression method. Returns scaled data, MSE- and R2-values. Is used when we do not apply the bootstrap and cross-validation resampling techniques, i.e. contains just a normal implementation.

## ols.py

Most of the analysis in regards to the OLS method.

## ridge.py

Most of the analysis in regards to the Ridge method.

## lasso.py

Most of the analysis in regards to the Lasso method.

## bootstrap.py

Contains the bootstrap algorithm, as well as some bootstrap analysis.

## cross_validation.py

Contains the k-fold cross-validation algorithm, as well as some cross-validation analysis.

