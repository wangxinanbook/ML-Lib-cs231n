'''
### Set up the environemt:
  Run in assignment/ source .env/bin/activate
### Turn off:
  deactivate
###
'''

# Run some setup code for this notebook.

import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
#from __future__ import print_function



# Load the raw CIFAR-10 data.
cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)


# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)


# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
#print(X_train.shape, X_test.shape)


'''
# Subsample the data for more efficient code execution in this exercise
num_training = 5000
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]

num_test = 500
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
'''

# Based on the cross-validation results above, choose the best value for k,   
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.

from cs231n.classifiers import KNearestNeighbor

# Create a kNN classifier instance. 
# Remember that training a kNN classifier is a noop: 
# the Classifier simply remembers the data and does no further processing 
classifier = KNearestNeighbor()


'''
# Distance distribution: Instrinsic difficulty of this data set
from numpy import histogram
classifier.train(X_train,y_train)
dists = classifier.compute_distances_no_loops(X_test)
#print(dists.shape)
#hist, bin_edges = np.histogram(dists.reshape((1,-1)), density=True)
plt.hist(dists.reshape(-1,1),bins=100,normed=True)
plt.title("Histogram for the distances.")
plt.show()
'''

def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    res = f(*args)
    toc = time.time()
    return toc - tic, res

'''
# Compare distance computation with  scipy

print('Training data shape: ', X_train.shape)
print('Test data shape: ', X_test.shape)

sp_L2_time,distA = time_function(cdist,X_test,X_train,'euclidean')
sp_L1_time,_ = time_function(cdist,X_test,X_train,'cityblock')

classifier.train(X_train,y_train)
vec_L2_time, distB = time_function(classifier.compute_distances_no_loops, X_test)

print("Comparing the time to calculate distances:")
print("\tScipy L2: ",sp_L2_time)
print("\tScipy L1: ",sp_L1_time)
print("\tVectorized L2: ",vec_L2_time)

difference = np.linalg.norm(distA-distB, ord='fro')
print('Difference was: %f' % (difference, ))
if difference < 0.001:
    print('Good! The distance matrices are the same')
else:
    print('Uh-oh! The distance matrices are different')

# Results (5000,3072) + (500,3072): Scipy L1,L2 time=>4s, vetorized L2: 0.2 
exit()
'''

### KNN Main Body #############################
import time
tic = time.time()

best_k = 10

classifier.train(X_train,y_train)

### L2 distance
# t_dist,dist_bst_v = time_function(classifier.compute_distances_no_loops,X_test)
### L1 distance
dist_bst_v = cdist(X_test,X_train,'cityblock')

#print("\tVectorized L2: ",t_dist)

#exit()

y_pred = classifier.predict_labels(dist_bst_v,best_k)

num_correct = np.sum(y_pred == y_test)
num_test = X_test.shape[0]
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

#raw_input("Press enter to continue")
#exit() 

toc = time.time()
print("Running time:", toc - tic)

### Data Set: CIFAR-10
###
### Running time: 181s, accuracy: 33.86%, L2 distance, (50000,3072) vs (10000,3072)
###   Distance calulation: 40s (Vectorized L2)
###
### Running time: 952s, accuracy: 38.1%, L1 distance, same data/test size
###############################################

