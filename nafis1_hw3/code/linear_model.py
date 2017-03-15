import numpy as np
from numpy.linalg import solve
import findMin
import sys
from scipy.optimize import approx_fprime


# Original Least Squares
class LeastSquares:
    # Class constructor
    def __init__(self):
        pass

    def fit(self,X,y):
        # Solve least squares problem

        a = np.dot(X.T, X)
        b = np.dot(X.T, y)
        self.w = solve(a, b)

    def predict(self, Xhat):

        w = self.w
        yhat = np.dot(Xhat, w)
        return yhat

# Least Squares with a bias added
class LeastSquaresBias:
    def __init__(self):
        pass

    def fit(self,X,y):
        num_rows, num_cols = X.shape
        N = num_rows
        x2 = np.ones((num_rows,1))
        Z = np.concatenate((X, x2), axis=1)        
        #Z = np.c_[X,np.ones(N)]

        #Solve least squares
        a = np.dot(Z.T, Z)
        b = np.dot(Z.T, y)
        self.w = solve(a, b)
        w = self.w


        ''' YOUR CODE HERE FOR Q2.1 '''
        # add a column of one to X
        # Solve least squares problem
        

    def predict(self, Xhat):

        ''' YOUR CODE HERE FOR Q2.1 '''
        w =self.w

        #add bias
        row1,col1 = Xhat.shape
        x2 = np.ones((row1,1))

        Z = np.concatenate((Xhat, x2), axis=1)        

        yhat = np.dot(Z,w)

        return yhat
        

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        #[n, d] = X.shape
        #solve least squares problem
        Z = self.__polyBasis(X)
        a = np.dot(Z.T, Z)
        b = np.dot(Z.T, y)
        self.w = solve(a,b)
       # w = self.w
      


        ''' YOUR CODE HERE FOR Q2.2 '''
        

    def predict(self, Xhat):
        w = self.w
        Z = self.__polyBasis(Xhat)
        yhat =  np.dot(Z,w)
        ''' YOUR CODE HERE FOR Q2.2 '''
        return yhat

    # A private helper function to transform any matrix X into 
    # the polynomial basis defined by this class at initialization 
    # Returns the matrix Z that is the polynomial basis of X.   
    def __polyBasis(self, X):
        
        n = X.shape[0]
        d = self.p + 1
        # Z should have as many rows as X and as many columns as (p+1)
        Z = np.ones((n, d))
        for i in range (self.p):
            Z[:,i+1]=(X**(i+1))[:,0]
        return Z

# Least Squares with RBF Kernel
class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self,X,y):
        self.X = X
        [n, d] = X.shape

        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        l = 1e-12

        a = Z.T.dot(Z) + l* np.identity(n)
        b = np.dot(Z.T, y)
        self.w = solve(a,b)

    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z.dot(self.w)
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma** 2))

        D = (X1**2).dot(np.ones((d, n2))) + \
            (np.ones((n1, d)).dot((X2.T)** 2)) - \
            2 * (X1.dot( X2.T))

        Z = den * np.exp(-1* D / (2 * (sigma**2)))
        return Z

# Least Squares where each sample point X has a weight associated with it.
class WeightedLeastSquares:

    def __init__(self):
        pass

    def fit(self,X,y,z):
        o=np.full((400,),1)
        p=np.full((100, ), .1)
        l = np.concatenate((o, p), axis=0)        
        #X.T*Z*Xw =X.T*Z*y)
        Z = np.diag(l)
       
        ''' YOUR CODE HERE FOR Q4.1 '''
        
        a = np.dot(X.T, Z)
        b = np.dot(a, X)
        b_in = np.linalg.inv(b)
        c = np.dot(b_in, a)
        self.w = (np.dot(c, y))

    def predict(self,Xhat):
        '''YOUR CODE HERE FOR Q4.1 '''
        w = self.w

        yhat = Xhat*w
        return yhat


class LinearModelGradient:

    def __init__(self):
        pass

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')
        

        self.w, f = findMin.findMin(self.funObj, self.w, 100, X, y)

    def predict(self,Xtest):

        w = self.w
        yhat = Xtest*w
        return yhat

    def funObj(self,w,X,y):

        ''' MODIFY THIS CODE FOR Q4.3 '''

        # Calculate the function value
        f=np.sum(np.log(np.exp(X*w-y)+np.exp(y-X*w)),axis=0)

        # Calculate the gradient value
        g = np.sum(X.T.dot((np.exp(X.dot(w)-y)-np.exp(y-X.dot(w)))/(np.exp(X.dot(w)-y)+np.exp(y-X.dot(w)))),axis=0)

        return (f,g)