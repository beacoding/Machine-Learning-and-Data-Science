"""
Implementation of c-nearest neighbours classifier
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
    
    N, D = X.shape 
   
    Xcond = X[0,None]
    
    ycond = y[0,None]
    ncond = 1
    
    for i in range (N):#go through subsequent training example 
            #y_pred = predict(X,Xtest)
        dist = utils.euclidean_dist_squared(Xcond,X[i,:])
        ds = np.argsort(dist, axis=0)
        y_pred =  stats.mode(ycond[ds[:min(k,ncond)]])[0][0]
    
        if y_pred != y[i]:#if the example is incorrectly classified by the KNN classifier using the current subset then
            Xcond = np.append(Xcond,X[i,None],axis=0)
            ycond = np.append(ycond,y[i,None],axis=0)    
            ncond = ncond + 1
            
        
    model = dict()
    model['X'] = Xcond
    model['y'] = ycond
    model['k'] = k
    model['predict'] = predict
    return model

def predict(model, Xtest):
    """ YOUR CODE HERE """
    X = model['X']
    y = model['y']
    k = model['k']

    T, D = Xtest.shape
    dist = utils.euclidean_dist_squared(X,Xtest)

    #dist[:,T]
    qw = np.argsort(dist, axis=0) # 
    y_pred = np.empty(T)#yhat = zeros(t ,1);
    for n in range (Xtest.shape[0]):
        y_pred[n] =  stats.mode(y[qw[:k,n]])[0][0]
        #[minDist,sorted] = sort(D(:,i));
        #yhat(i) = mode(y(sorted(1:k)));
    return y_pred
   

   