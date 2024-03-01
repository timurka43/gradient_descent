"""
Title: Homework 2, part 1. Gradient Descent
Author: Timur Kasimov (Grinnell, 2025)
Data Created: February 20, 2024
Date Updated: March 1st, 2024

Purpose: Implement and test a gradient descent method

This is a homework assignment for my Computational Methods Class
"""

import numpy as np


MAXITER = 1000
TOLERANCE = 1e-10

# matrix and vector for part 2 (optimal step alpha)
A = np.array([2,-1,0,-1,2,-1,0,-1,2]).reshape(3, 3)
# print(A)
b = np.array([2,0,2])
# print(b)

# constants for part 3 (backtracking method)
RHO = 0.9
c = 0.1


########################
### GRADIENT DESCENT ###
########################
'''
Pre-conditions:
  f: function of n variables
  gradient df: a function that computes the gradient of f at given coordinates
  guess: vector of size n with initial guess coordinates
  alpha: a function that calculates optimal step, or a constant step size
  backtracking: if True, use backtracking method instead of constant/optimal alpha
Post-conditions: 
  returns vector of size n with a coordinates of the min of f
'''
def gradient_descent(f, df, guess, alpha, backtracking = False):
    # iterative process, stopping condition: reached max iterations (MAXITER)
    for i in range(0, MAXITER):
        # Check if the current guess is already at a min.
        # For this, the norm of gradient vector needs to be zero (within tolerance)
        gradient_vector = df(guess)
        if (np.linalg.norm(gradient_vector) < TOLERANCE):
            print("Iterations: ", i)
            print("Minimum at: ", np.round(guess, 5))
            print()
            return np.round(guess, 5) # return the guess vector that evaluates to the min of the function 
        # if did not find the min, first decide on the step size
        # if backtracking=True, reduce step by constant factor until Wolfe condition is met
        if (backtracking):
            step = alpha
            # shrink step until Wolfe condition is met
            while (f(guess - step * gradient_vector) > (f(guess) - c*step* (gradient_vector@gradient_vector))):
                step = RHO * step
        # else if not backtracking but alpha is given as a constant step, use it directly
        elif (type(alpha) == float):
            step = alpha
        # if alpha is a function, compute the optimal step
        else:
            step = alpha(gradient_vector) 
            # print("Step: ", step) # debugging
        # after determining the step size, calculate the new guess
        guess1 = guess - step * gradient_vector 
        # print("New guess", guess1) # debugging
        # stopping condition: if difference between guesses is negligible
        if (np.linalg.norm(guess1-guess) < TOLERANCE):
            print("Difference between guesses is negligible")
            print("Iterations: ", i)
            print("Minimum at: ", np.round(guess1, 5))
            print()
            return np.round(guess1, 5)
        # update guess, repeat loop
        guess = guess1
    # if reached max iterations:
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

# # Debugging 
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

# # Debugging
# print("CHECK gradient f4")
# gradf4(np.array([-1,0,1]))
# gradf4(np.array([1,0,1])) # works correctly





'''
calculating optimal step size for certain objective functions (f4)
'''
def get_alpha(gradient):
    return (gradient@gradient)/((gradient@A)@gradient)

# # Debugging
# print("Check get_alpha")
# print(get_alpha(np.array([1,2,3]))) # works correctly





if __name__ == "__main__":

    # part 1
    print("Part 1")
    print("Function 1")
    
    myguess = np.array([1.0, 1.0])
    gradient_descent(f1, gradf1, myguess, .9) # large step, slow convergence
    print()
    print()
    gradient_descent(f1, gradf1, myguess, .1) # small step, also slow 
    print()
    print()
    gradient_descent(f1, gradf1, myguess, .4) # much better middle ground, only 15 iterations
    print()
    print()
    gradient_descent(f1, gradf1, myguess, .5) # single iteration, lucky choice of step
    print()
    print()



    print("Function 2")
    myguess1 = np.array([2.0, 1.0])
    gradient_descent(f2, gradf2, myguess1, 0.9) # overflow, diverges
    gradient_descent(f2, gradf2, myguess1, 0.0000001) # convergence is too slow
    ''' Either diverges (if step is too big) or converges extremely slowly.

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
    ''' all guesses work correctly, produce [2,2,2] as an answer'''



    # part 3
    print()
    print("Part 3")

    print("Function 1")
    gradient_descent(f1, gradf1, myguess, 1.0, backtracking=True)
    ''' correct here '''

    print("Function 2")
    gradient_descent(f2, gradf2, myguess, 1.0, backtracking=True)
    ''' same issue with poorly scaled issues as with constant step'''

    print("Function 3")
    gradient_descent(f3, gradf3, np.array([10,10,10,10,10]), 1.0, backtracking=True)
    ''' works fine here too'''


