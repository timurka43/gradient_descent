"""
Title: Homework 2, part 1. Gradient Descent
Author: Timur Kasimov (Grinnell, 2025)
Data Created: February 20, 2024
Date Updated: February 20, 2024

Purpose: Implement and test a gradient descent method

This is a homework assignment for my Computational Methods Class
"""

import math
import numpy as np

########################
### GRADIENT DESCENT ###
########################
'''
Pre-conditions:
  f: function of n variables
  gradient: a vector with the gradient of function f
  guess: vector of size n with initial guess coordinates
Post-conditions: 
  returns vector of size n with a coordinates of the min/max of f
Description: 
'''
def gradient_descent(f, gradient, guess):
    return



########################
###  MATH FUNCTION   ###
########################
def f1(x1,x2):
    return x1*x1 + x2*x2

def gradf1(x1, x2):
    return np.array([2*x1, 2*x2])