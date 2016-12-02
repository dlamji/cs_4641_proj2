"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import sklearn
import numpy as np
import numpy.linalg as la
import math


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    '''
        Arguments:  
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''
    n1 = len(X1)
    n2 = len(X2)
    ker = (np.dot(X1,X2.T)+1)**_polyDegree
    return ker




def myGaussianKernel(X1, X2):
    '''
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    '''


    n1 = len(X1)
    n2,d = X2.shape
    ker = None
    AA = (X1**2).dot(np.ones((d,n2)))
    BB = np.ones((n1,d)).dot((X2.T**2))

    ker = -2*X1.dot(X2.T) + AA + BB
    ker = np.exp((-ker/(2*_gaussSigma)))
    return ker
