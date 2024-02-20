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


MAXITER = 1000
TOLERANCE = 1e-10

########################
### GRADIENT DESCENT ###
########################
'''
Pre-conditions:
  f: function of n variables
  gradient: a function that computes the gradient of f at given coordinates
  guess: vector of size n with initial guess coordinates
Post-conditions: 
  returns vector of size n with a coordinates of the min/max of f
Description: 
'''
def gradient_descent(f, get_gradient, guess, alpha):
    for i in range(1, MAXITER+1):
        # Check if the current guess is already at a min.
        # For this, the gradient vector needs to be a zero vector.
        gradient_vector = get_gradient(guess)
        if (np.all(abs(gradient_vector) <= TOLERANCE)):
            print("Iterations: ", i)
            print("Minimum at: ", guess)
            return guess # return the guess vector that evaluates to the min of the function 
        # compute the new guess
        guess = guess - alpha * gradient_vector
        print(guess)

    print("MAX ITERATIONS REACHED")
    print("Closest approximation of the minimum at: ", guess)
    return guess



########################
###  MATH FUNCTION   ###
########################
def f1(vec):
    x1 = vec[0]
    x2 = vec[1]
    return x1*x1 + x2*x2

def gradf1(vec):
    x1 = vec[0]
    x2 = vec[1]
    return np.array([2*x1, 2*x2])


if __name__ == "__main__":
    myguess = np.array([2.0, 10.0])
    gradient_descent(f1, gradf1, myguess, 0.1)