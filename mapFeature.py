import numpy as np

def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    n = len(x1)
    featureMatirx = np.zeros((n,28))
    for row in range(n):
        count = 0
        for i in range (7):
            for j in range (i+1):
                exp2 = j
                exp1 = i-j
                featureMatirx[row][count] = (x1[row]**exp1) * (x2[row]**exp2)
                count += 1
    return featureMatirx

