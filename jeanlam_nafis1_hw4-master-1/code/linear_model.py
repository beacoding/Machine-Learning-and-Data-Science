from __future__ import division

import numpy as np
import minimizers
import utils

class logReg:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100):
        self.verbose = verbose
        self.maxEvals = maxEvals

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape    

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMin(self.funObj, self.w, 
                                         self.maxEvals, 
                                         self.verbose,
                                         X, y)
    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL1:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100, lammy=1):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = lammy
    def funObj(self, w, X, y):
        
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g

    def fit(self,X, y):
        n, d = X.shape    

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMinL1(self.funObj, self.w, self.lammy,
                                         self.maxEvals, 
                                         self.verbose,
                                         X, y)
    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)

class logRegL2:
    # Logistic Regression
    def __init__(self, verbose=1, maxEvals=100, lammy=1):
        self.verbose = verbose
        self.maxEvals = maxEvals
        self.lammy = lammy
    def funObj(self, w, X, y):
        
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw))) + (self.lammy/2 * w.T.dot(w))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res) + self.lammy*w

        return f, g

    def fit(self,X, y):
        n, d = X.shape    

        # Initial guess
        self.w = np.zeros(d)
        utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMin(self.funObj, self.w, 
                                         self.maxEvals, 
                                         self.verbose,
                                         X, y)
    def predict(self, X):
        w = self.w
        yhat = np.dot(X, w)

        return np.sign(yhat)


# L0 Regularized Logistic Regression
class logRegL0(logReg): # this is class inheritance:
    # we "inherit" the funObj and predict methods from logReg
    # and we overwrite the __init__ and fit methods below.
    # Doing it this way avoids copy/pasting code. 
    # You can get rid of it and copy/paste
    # the code from logReg if that makes you feel more at ease.
    def __init__(self, L0=1.0, verbose=2, maxEvals=400):
        self.verbose = verbose
        self.L0 = L0
        self.maxEvals = maxEvals

    def fit(self, X, y):
        n, d = X.shape    
        w0 = np.zeros(d)
        minimize = lambda ind: minimizers.findMin(self.funObj, 
                                                  w0[ind], 
                                                  self.maxEvals, 0, 
                                                  X[:, ind], y)
        selected = set()
        selected.add(0) # always include the bias variable 
        minLoss = np.inf
        oldLoss = 0
        bestFeature = -1

        minScore = np.inf
        minIndex = -1
        
        # ignore = false
        while minLoss != oldLoss:
            oldLoss = minLoss
            if self.verbose > 1:
                print("Epoch %d " % len(selected))
                print("Selected feature: %d" % (bestFeature))
                print("Min Loss: %.3f\n" % minLoss)

            for i in range(d):
                if i in selected:
                    continue
                
                selected_new = selected | {i} # add "i" to the set
                
                new_w, value = minimize(list(selected_new))
                # print value
                if value < minScore:
                    minScore = value
                    minIndex = i
                # TODO: Fit the model with 'i' added to the features,
                # then compute the score and update the minScore/minInd
            
            
            selected.add(minIndex)
            minLoss = minScore
        
        # re-train the model one last time using the selected features
        self.w = w0
        self.w[list(selected)], _ = minimize(list(selected))       


class leastSquaresClassifier:
    # Q3 - One-vs-all Least Squares for multi-class classification
    def __init__(self):
        pass

    def fit(self, X, y):
        n, d = X.shape    
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))
        
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1


            self.W[:, i] = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, ytmp))[0]

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)


class logLinearClassifier:
    # Q3 - One-vs-all Least Squares for multi-class classification
    def __init__(self, maxEvals, verbose):
        self.maxEvals = maxEvals
        self.verbose = verbose

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)

        # Calculate the function value
        f = np.sum(np.log(1. + np.exp(-yXw)))

        # Calculate the gradient value
        res = - y / (1. + np.exp(yXw))
        g = X.T.dot(res)

        return f, g


    def fit(self, X, y):
        n, d = X.shape    
        self.n_classes = np.unique(y).size

        # Initial guess
        self.W = np.zeros((d, self.n_classes))
        
        for i in range(self.n_classes):
            ytmp = y.copy().astype(float)
            ytmp[y==i] = 1
            ytmp[y!=i] = -1

            # self.W[:, i] = np.linalg.lstsq(np.dot(X.T, X), np.dot(X.T, ytmp))[0]
            self.w = np.zeros(d)
            
            self.W[:, i] = minimizers.findMin(self.funObj, self.w, 
                                         self.maxEvals, 
                                         self.verbose,
                                         X, ytmp)[0]

    def predict(self, X):
        yhat = np.dot(X, self.W)

        return np.argmax(yhat, axis=1)


class softmaxClassifier:
    # Q3 - One-vs-all Least Squares for multi-class classification
    def __init__(self, maxEvals, verbose):
        self.maxEvals = maxEvals
        self.verbose = verbose

    def funObj(self, w, X, y):
        n, d = X.shape
        
        self.k = np.unique(y).size
        self.W = np.reshape(w, (d, self.k))

        print "X shape:", X.shape
        print "k", self.k
        print "W shape", self.W.shape

        f = 0
        for index in range(n):
            yVal = y[index]
            value = X[index].dot(w[yVal]) * -1
            logsum = 0
            for c in range(self.k):
                logsum += np.exp(w[c] * (X[index]))
            f = f + value + np.log(logsum)
      
        g = np.zeros((d,self.k))
        # val= np.exp(X.dot(self.W)).shape
        probability = np.exp(X.dot(self.W))/np.sum(np.exp(X.dot(self.W)), axis=1, keepdims=True)
        print "Probability", probability.shape
        print "len(x)", len(X)
        for j in range(d):
            val = np.zeros((n,self.k))
            for i in range(len(probability)):
                for c in range(self.k):
                    # print "prob[i]", probability[i]
                    # print "xij", X[i,j]
                    # print "y==", y[i]==c
                    val[i] = probability[i]*X[i,j] - (X[i, j]*(y[i]==c))
            print val
            print "*************"
            print np.sum(val, axis=0).shape
            g[j,] = np.sum(val, axis=0)
     
                    
        print g.ravel().shape
        return f, g.ravel()


    def fit(self, X, y):
        n, d = X.shape    
        self.n_classes = np.unique(y).size
        self.k = np.unique(y).size
        # Initial guess
        self.w = np.zeros(d*self.k)
        # utils.check_gradient(self, X, y)
        (self.w, f) = minimizers.findMin(self.funObj, self.w, 
                                         self.maxEvals, 
                                         self.verbose,
                                         X, y)
        
        self.w = np.reshape(self.w, (d, k))

    def predict(self, X):
        yhat = np.dot(X, self.w)

        return np.argmax(yhat, axis=1)