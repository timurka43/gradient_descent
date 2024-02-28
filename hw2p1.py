"""
Title: Homework 2, part 1. Gradient Descent
Author: Timur Kasimov (Grinnell, 2025)
Data Created: February 20, 2024
Date Updated: February 27, 2024

Purpose: Implement and test a gradient descent method

This is a homework assignment for my Computational Methods Class
"""

import math
import numpy as np


MAXITER = 1000
TOLERANCE = 1e-10

A = np.array([1,0,3,2,2,2,0,1,2]).reshape(3, 3)
# print(A)
b = np.array([2,0,2])
# print(b)

RHO = 0.9
c = 0.1


########################
### GRADIENT DESCENT ###
########################
'''
Pre-conditions:
  f: function of n variables
  gradient: a function that computes the gradient of f at given coordinates
  guess: vector of size n with initial guess coordinates
  alpha: constant step size
Post-conditions: 
  returns vector of size n with a coordinates of the min of f
'''
def gradient_descent(f, df, guess, alpha, backtracking = False):
    for i in range(1, MAXITER+1):
        # Check if the current guess is already at a min.
        # For this, the gradient vector needs to be a zero vector (within tolerance)
        gradient_vector = df(guess)
        if (np.linalg.norm(gradient_vector) < TOLERANCE):
            print("Iterations: ", i)
            print("Minimum at: ", np.round(guess, 5))
            print()
            return np.round(guess, 5) # return the guess vector that evaluates to the min of the function 
        # compute the new guess
        # if backtracking, reduce step until Wolfe condition is met
        if (backtracking):
            step = alpha
            # shring step until Wolfe condition is met
            while (f(guess - step * gradient_vector) > (f(guess) - c*step* (gradient_vector@gradient_vector))):
                step = RHO * step
        # else if not backtracking but alpha is given as a constant step, use it directly
        elif (type(alpha) == float):
            step = alpha
        # if alpha is a function, compute the optimal step
        else:
            step = alpha(gradient_vector)/10 #smth is wrong with the step calculation here. it is too big. if dividing step by 10, it converges. 
            # print("Step: ", step) # debugging
        guess1 = guess - step * gradient_vector 
        # print("New guess", guess1) # debugging
        if (np.linalg.norm(guess1-guess) < TOLERANCE):
            print("Difference between guesses is negligible")
            print("Iterations: ", i)
            print("Minimum at: ", np.round(guess1, 5))
            print()
            return np.round(guess1, 5)
        guess = guess1
    print("MAX ITERATIONS REACHED")
    print("Closest approximation of the minimum at: ", guess)
    print()
    return np.round(guess, 5)



########################
###  MATH FUNCTIONS  ###
########################
def f1(vec):
    x1 = vec[0]
    x2 = vec[1]
    return x1*x1 + x2*x2

def f2(vec):
    x1 = vec[0]
    x2 = vec[1]
    return (10**6) * x1*x1 + x2*x2


def f3(vec):
    x1 = vec[0]
    x2 = vec[1]
    x3 = vec[2]
    x4 = vec[3]
    x5 = vec[4]
    return x1*x1 + x2*x2 + x3*x3 + x4*x4 + x5*x5 


def f4(vec):
    return 0.5 * ((vec@A)@vec) - b@vec
    
# print("CHECK f4")
# print(f4(np.array([-1,0,1])))  # works correctly



########################
###    GRADIENTS     ###
########################

def gradf1(vec):
    x1 = vec[0]
    x2 = vec[1]
    return np.array([2*x1, 2*x2])

def gradf2(vec):
    x1 = vec[0]
    x2 = vec[1]
    return np.array([2*(10**6)*x1, 2*x2])

def gradf3(vec):
    x1 = vec[0]
    x2 = vec[1]
    x3 = vec[2]
    x4 = vec[3]
    x5 = vec[4]
    return np.array([2*x1, 2*x2, 2*x3, 2*x4, 2*x5])

def gradf4(vec):
    gradient = A@vec - b
    # print("Gradient")
    # print(gradient)
    return gradient

# print("CHECK gradient f4")
# gradf4(np.array([-1,0,1]))
# gradf4(np.array([1,0,1])) # works correctly





'''
calculating optimal step size for certain objective functions
'''
def get_alpha(gradient):
    return (gradient@gradient)/((gradient@A)@gradient)

# print("Check get_alpha")
# print(get_alpha(np.array([1,2,3]))) # works correctly





if __name__ == "__main__":

    # part 1
    print("Part 1")
    print("Function 1")
    
    myguess = np.array([1.0, 1.0])
    gradient_descent(f1, gradf1, myguess, .9)
    print()
    print()
    gradient_descent(f1, gradf1, myguess, .1)
    print()
    print()
    gradient_descent(f1, gradf1, myguess, .4)
    print()
    print()
    gradient_descent(f1, gradf1, myguess, .5)
    print()
    print()

    # print("Function 2")
    # myguess1 = np.array([2.0, 1.0])
    # # gradient_descent(f2, gradf2, myguess1, 0.9) #overflow, need to scale??
    # # gradient_descent(f2, gradf2, myguess1, 0.0000001)
    ''' Does not converge well. If we choose a big constant step, 

    The scaling is so bad that the method with constant step size does
    not work here. Poorly behaved.
    '''




    # part 2
    print()
    print("Part 2")


    guess2 = np.array([-1.0, 0.0, 1.1])
    guess3 = np.array([0.0, 0.0, 0.0])
    guess4 = np.array([10.0, 10.0, 10.0])
    gradient_descent(f4, gradf4, guess2, get_alpha)
    print()
    print()
    gradient_descent(f4, gradf4, guess3, get_alpha)
    print()
    print()
    gradient_descent(f4, gradf4, guess4, get_alpha)
    '''
    The optimal step seems wrong. function only converges when we divide alpha by about 1.
    Diverges if alpha is kept according to the formula
    '''


    # part 3
    print()
    print("Part 3")

    print("Function 1")
    gradient_descent(f1, gradf1, myguess, 1.0, backtracking=True)

    print("Function 2")
    gradient_descent(f2, gradf2, myguess, 1.0, backtracking=True)

    print("Function 3")
    gradient_descent(f3, gradf3, np.array([10,10,10,10,10]), 1.0, backtracking=True)


