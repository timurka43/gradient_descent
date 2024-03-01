"""
Title: Homework 2, part 2. Newton's Method in Multiple Dimensions
Author: Timur Kasimov (Grinnell, 2025)
Data Created: February 27, 2024
Date Updated: March 1st, 2024

Purpose: Implement and test Newton's method in Multiple Dimensions

This is a homework assignment for my Computational Methods Class
"""

import math
import numpy as np
import matplotlib.pyplot as plt


MAXITER = 1000
TOLERANCE = 1e-10


#####################
## NEWTON'S METHOD ##
#####################
'''
Pre-conditions:
  f: function of n variables
  gradient df: a function that computes the gradient of f at given coordinates
  hesssian: a function that takes in a guess vector and computes hessian matrix at that position
  guess: vector of size n with initial guess coordinates
Post-conditions: 
  returns list of vectors of size n with all the guesses
  prints the best guess for the minimum of the function f
'''
def newton(f, df, hessian, guess):
    # initialize a list ot record all guesses that will be returned
    guesses = []
    # append initial guess
    guesses.append(guess)
    # run loop until reached max iterations
    for i in range(0, MAXITER):
        # Check if the current guess is already at a min.
        # For this, the norm of the gradient vector needs to be zero (within tolerance)
        gradient_vector = df(guess)
        if (np.linalg.norm(gradient_vector) < TOLERANCE):
            print("Iterations: ", i)
            print("Minimum at: ", np.round(guess, 5))
            print()
            return guesses # return the guess vector that evaluates to the min of the function 
        # compute the new guess, update
        guess1 = guess - np.linalg.inv(hessian(guess)) @ gradient_vector
        #append the new guess right awayy
        guesses.append(guess1) 
        #if the new guess is not different from prev guess (within tolerance) , then terminate
        if (np.linalg.norm(guess1-guess) < TOLERANCE):
            print ("Iterations:", i)
            print("Zero at x≈", np.round(guess1, 5))
            return guesses
        else:
            # update the current guess, iterate again
            guess = guess1
    # stopping condition: reached max num of iterations
    print("REACHED MAXIMUM NUMBER OF ITERATIONS")
    print("The closest approximation of zero is at x=", round(guess, 6))
    return guesses


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
    x1 = vec[0]
    x2 = vec[2]
    return math.cos(x1) + math.sin(x2)



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
    x1 = vec[0]
    x2 = vec[1]
    return np.array([-1 * math.sin(x1), math.cos(x2)])



########################
###     HESSIANS     ###
########################

def Hf1(vec): 
    return np.array([2,0,0,2], dtype="float").reshape(2,2)

def Hf2(vec):
    return np.array([2*(10**6),0,0,2], dtype="float").reshape(2,2)

def Hf3(vec):
    return np.array([[2,0,0,0,0],
                     [0,2,0,0,0],
                     [0,0,2,0,0],
                     [0,0,0,2,0],
                     [0,0,0,0,2]], dtype="float")

def Hf4(vec):
    x1 = vec[0]
    x2 = vec[1]
    return np.array([[-1 * math.cos(x1), 0],
                     [0, -1 * math.sin(x2)]], dtype="float")





#########################################
####     FUNCTION TO PLOT ERRORS     ####
#########################################
def plot_errors(guesses, x_bar_vec, title):
    errors = []
    for vec_guess in guesses:
        errors.append(np.linalg.norm(x_bar_vec - vec_guess))
        # print(errors)
    # plotting all but the last errors on x-axis, and all but the first errors on y-axis
    plt.loglog(errors[:-1], errors[1:])
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # Part 1
    print("Part 1")
    print("Function 1")
    
    newton(f1, gradf1, Hf1, np.array([10.0, 15.0]))

    print("Function 2")
    newton(f2, gradf2, Hf2, np.array([100.0, 150.0]))
    
    print("Function 3")
    newton(f3, gradf3, Hf3, np.array([100.0, 150.0, -94.0, 17.234, 0.0]))
    print()

    print("Function 4")
    newton_guesses_4 = newton(f4, gradf4, Hf4, np.array([3.0, 4.0]))
    print()
    print()


    #Part 2
    '''
    Newton method seemingly only takes 1 iteration, or exactly one update.
    This is significantly smaller than the number of iterations that
    gradient descent takes, specifically between 1 and 108 for function 1,
    depending on how good the initial guess is.

    Gradient descent does not converge for function 2 due to scaling, 
    or converges extremely slowly if we take small enough steps. However,
    newton's method converges in 2 iterations for function 2, so it does not face
    issues with poorly scaled problems.


    UPDATE:
    once we have a more complicated function, like f4 above, newton method takes more iterations 
    to converge, since our derivatives are no longer linear. 
    Specifically, function 4 takes 4 iterations to converge when the initial guess is [3,4]
     '''
    


    # Part 3
    print("Part 3")
    print(newton_guesses_4)
    plot_errors(newton_guesses_4, np.array([math.pi, 3*math.pi/2]), "Convergence Rate")
    ''' 
    The plot definitely has at least quadratic convergence, but perhaps even better?
    The slope to me looks like it is closer to 3 than it is to 2.
    '''
