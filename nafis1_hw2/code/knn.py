"""
Implementation of k-nearest neighbours classifier
"""
 
import numpy as np
from scipy import stats
import utils

def fit(X, y, k):
    """
    Parameters
    ----------
    X : an N by D numpy array
    y : an N by 1 numpy array of integers in {1,2,3,...,c}
    k : the k in k-NN
    """
    # Just memorize the training dataset
    model = dict()
    model['X'] = X
    model['y'] = y
    model['k'] = k
    model['predict'] = predict
    return model

def predict(model, Xtest):
    """ YOUR CODE HERE """
    X = model['X']
    y = model['y']
    k = model['k']

    T, D = Xtest.shape
    N, D = X.shape
    dist = utils.euclidean_dist_squared(X,Xtest)

    qw = np.argsort(dist, axis=0) 
    y_pred = np.empty(T)
    for n in range (T):
        y_pred[n] =  stats.mode(y[qw[:k,n]])[0][0]
    return y_pred
   

   