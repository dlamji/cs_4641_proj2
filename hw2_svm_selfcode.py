"""
=====================================
Test SVM with custom Gaussian kernels
=====================================

"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import cross_validation
import svmKernels
from sklearn.metrics import accuracy_score


# import some data to play with
filename = 'data/svmTuningData.dat'
allData = np.loadtxt(filename, delimiter=',')
X = allData[:, :-1]         # Indexes from the end of the array.
Y = allData[:,  -1]

print "Training the SVMs..."

k_iters = 10
accuracy = np.zeros(k_iters)
best_accuracy = 0
best_sigma = 0

# 20.48 for Sigma And 650 for C
# C:  636 Sigma:  21.0
# Accuracy:  0.978260869565
C = 500
svmKernels._gaussSigma = 13.3

for k in range(0, k_iters):
  X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.35, random_state=0)
  
  # create an instance of SVM with the custom kernel and train it
  myModel = svm.SVC(C = C, kernel=svmKernels.myGaussianKernel)
  myModel.fit(X_train, y_train)
  
  # Predict results with the test data
  y_testpred = myModel.predict(X_test)
  # Compute Accuracy for this cross validation
  accuracy[k] = accuracy_score(y_test, y_testpred)
      
  accuracy_loop = np.mean(accuracy)
  if(accuracy_loop > best_accuracy):
    best_accuracy = accuracy_loop
    best_C = C
    best_sigma = svmKernels._gaussSigma
    

C = 636 # best_C
svmKernels._gaussSigma = 21.0 # best_sigma
print "C: ", C, "Sigma: ", svmKernels._gaussSigma
print "Accuracy: ", best_accuracy

# create an instance of SVM with the custom kernel and train it
myModel = svm.SVC(C = C, kernel=svmKernels.myGaussianKernel)
myModel.fit(X, Y)

# create an instance of SVM with build in RBF kernel and train it
equivalentGamma = 1.0 / (2 * svmKernels._gaussSigma ** 2)
model = svm.SVC(C = C, kernel='rbf', gamma=equivalentGamma)
model.fit(X, Y)

print ""
print "Testing the SVMs..."

h = .02  # step size in the mesh

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



# get predictions for both my model and true model
myPredictions = myModel.predict(np.c_[xx.ravel(), yy.ravel()])
myPredictions = myPredictions.reshape(xx.shape)

predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
predictions = predictions.reshape(xx.shape)

# plot my results
plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, myPredictions, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired) # Plot the training points
plt.title("SVM with My Custom Gaussian Kernel (sigma = "+str(svmKernels._gaussSigma) + ", C = "+str(C)+")")
plt.axis('tight')

# plot built-in results
plt.subplot(1, 2, 2)
plt.pcolormesh(xx, yy, predictions, cmap=plt.cm.Paired)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired) # Plot the training points
plt.title('SVM with Equivalent Scikit_learn RBF Kernel for Comparison')
plt.axis('tight')

plt.show()