'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''
import math
import copy
import numpy as np
import random


class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.regLambda = regLambda
        self.epsilon = epsilon
        self.maxNumIters = maxNumIters
        self.theta = None
    

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
    	n,d = X.shape
    	costsum = 0
    	for i in range (n):
    		thetaTx = theta.T.dot(X[i])[0]
    		#print self.sigmoid(thetaTx), (1-y[i])
    		if(y[i] == 0):
    			costsum += -math.log(1-self.sigmoid(thetaTx))
    		else:
    			costsum += -math.log(self.sigmoid(thetaTx))

    	thetasum = long(0)
    	for i in range (d):
    		thetasum += (theta[i])**2

    	thetasum = long(((thetasum) ** 0.5)*regLambda/2)
    	return costsum + thetasum
    
    def hasConverged(self, thetanew, thetaold, epsilon):

    	n = len(thetanew)
    	diffsum = 0
    	for i in range (n):
    		diffsum += (thetanew[i] - thetaold[i])**2
    	diffsum = diffsum ** 0.5
    	if(diffsum <= epsilon):
    		return True
    	return False
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
    	n,d = X.shape
    	gradientsum = 0
    	oldtheta = copy.deepcopy(theta)
    	newtheta = copy.deepcopy(theta)
    	for i in range(n):
    		thetaTx = oldtheta.T.dot(X[i])[0]
    		gradientsum += (self.sigmoid(thetaTx)-y[i])
    	newtheta[0] = gradientsum
    	for j in range(1,d):
    		gradientsum = 0
    		for i in range(n):
    			thetaTx = oldtheta.T.dot(X[i])[0]
    			gradientsum += (self.sigmoid(thetaTx)-y[i])*X[i][j]
    		newtheta[j] = gradientsum + regLambda*oldtheta[j]
    	return newtheta


    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        # initialize theta for gradient descent
        X = np.c_[np.ones((len(X),1)), X]
        if(self.theta == None):
        	d = len(X[0])
        	self.theta = np.zeros((d,1))
        	sumofprev = 0
        	for i in range(d-1):
        		self.theta[i] = random.uniform(-2,2)
        		sumofprev += self.theta[i]
        	self.theta[d-1] = -sumofprev


        n,d = X.shape
        for i in xrange (self.maxNumIters):
        	# print self.computeCost(self.theta, X, y, self.regLambda)
        	oldtheta = copy.deepcopy(self.theta)
        	gradient = self.computeGradient(oldtheta,X,y,self.regLambda)        	
        	self.theta = oldtheta - self.alpha*gradient
        	#print self.theta,oldtheta
        	if(self.hasConverged(oldtheta,self.theta,self.epsilon)):
        	#print "Totally",i,"number of iterations\n"
        		return
        	
        # logX = self.sigmoid(X)
        # self.theta = np.linalg.inv(logX.T.dot(logX)).dot(logX.T).dot(y)


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        X = np.c_[np.ones((len(X),1)),X]
        y = X.dot(self.theta)
        tmpy = self.sigmoid(y)
        for i in range(len(y)):
        	if(tmpy[i] >= 0.5):
        		y[i] = 1
        	else:
        		y[i] = 0 
        return y



    def sigmoid(self, z):
    	return 1 / (1 + np.exp(-z))
    	if(type(z) is np.float64):
    		return 1 / (1 + np.exp(-z))
    	# z is a n-by-d numpy matrix 
    	n,d = z.shape
    	sigmoidz = np.zeros((n,d))

    	for i in range (n):
    		for j in range (d):
    			sigmoidz[i][j] = 1 / (1 + np.exp(-z[i][j]))

    	return sigmoidz

    	
