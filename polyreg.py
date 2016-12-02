'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO
        self.degree = degree
        self.regLambda = regLambda
        self.theta = np.zeros((self.degree+1,1))
        self.mean = None
        self.std = None


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        d = self.degree
        n = len(X)
        polyFeat = np.zeros((n,degree))
        for i in range(n):
            for j in range(1,degree+1):
                polyFeat[i][j-1] = X[i] ** j
        return polyFeat
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #TODO
        #standardize the data
        
        polyX = self.polyfeatures(X,self.degree)
        n,d = polyX.shape
        self.mean = np.apply_along_axis(getMean,0,polyX)
        self.std = np.apply_along_axis(getStd,0,polyX)

        polyX = self.standardize(polyX)
        # add the x_0 column
        polyX = np.c_[np.ones((n,1)),polyX]
        n,d = polyX.shape
        regMatrix = self.regLambda * np.eye(d)
        regMatrix[0,0] = 0
        # try:
        #     self.theta = np.linalg.inv(polyX.T.dot(polyX)+regMatrix).dot(polyX.T).dot(y)
        # except np.linalg.linalg.LinAlgError:
        self.theta = np.linalg.pinv(polyX.T.dot(polyX)+regMatrix).dot(polyX.T).dot(y)
         
    
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        
        polyX = self.polyfeatures(X,self.degree)
        n,d = polyX.shape
        polyX = self.standardize(polyX)
        # add the x_0 column    
        polyX = np.c_[np.ones((n,1)),polyX]
        return polyX.dot(self.theta)

    def standardize(self,matrix):
        # print n,d
        n,d = matrix.shape        
        stdmtx = np.zeros((n,d))
        for i in range(d):
            meani = np.ones((d,1))*self.mean[i]
            stdi = np.ones((d,1))*self.std[i]
            stdmtx[:,i] = (matrix[:,i] - meani[0])/stdi[0]
        return stdmtx


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------
def getMean(a):
    return np.mean(a)

def getStd(a):
    return np.std(a)


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrain -- errorTrain[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTest -- errorTrain[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain)
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    #TODO -- complete rest of method; errorTrain and errorTest are already the correct shape
    
    pr_model = PolynomialRegression(degree, regLambda)

    for i in range(3,n+1):
        sumoftrainerror = 0
        tmpXtrain = Xtrain[0:i]
        tmpYtrain = Ytrain[0:i]
        pr_model.fit(tmpXtrain,tmpYtrain)
        Ytestpredict = pr_model.predict(Xtest)
        Ytrainpredict = pr_model.predict(tmpXtrain)

        for j in range(i):
            sumoftrainerror += (Ytrainpredict[j] - Ytrain[j]) ** 2
        sumoftrainerror /= i

        errorTest[i-1] = (Ytestpredict[0] - Ytest[0]) ** 2
        errorTrain[i-1] = sumoftrainerror
        # if(degree==8 and regLambda==0):
        #     print "Ytrain:",Ytrainpredict
    

    return (errorTrain, errorTest)
