
import sys
import argparse
import linear_model
import matplotlib.pyplot as plt
import numpy as np
import utils
import os

if __name__ == "__main__":
    argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required = True, 
        choices = ["2.1", "2.2","3.1","4.1","4.3"])
    io_args = parser.parse_args()
    question = io_args.question
    
    
    if question == "2.1":
        # Load the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        
        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]
        
        # Fit least-squares model
        model = linear_model.LeastSquaresBias()
        model.fit(X,y)
        
        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2) / n
        print("Training error = ", trainError)
        
        # Compute test error
        
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2) / t
        print ("Test error = ", testError)
        
        # Plot model
        plt.figure()
        plt.plot(X,y,'b.', label = "Training data")
        plt.title('Training Data')
        # Choose points to evaluate the function
        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g', label = "Least squares fit")
        plt.legend(loc="best")
        figname = os.path.join("..","figs","leastSquares.pdf")
        print("Saving", figname)
        plt.savefig(figname)
        
        
        ''' YOUR CODE HERE'''
        # Fit the least squares model with bias
        # Compute training error
        # Compute test error
        # Plot model
        # Choose points to evaluate the function
        
    elif question == "2.2":
        
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        
        # get the number of rows(n) and columns(d)
        n,d = X.shape
        t = Xtest.shape[0]
        
        
        for p in range(11):
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            yhat = model.predict(X)
            trainError = np.sum((yhat - y)**2) / n
            print("P = ", p)

            print("Training error = ", trainError)
            ''' YOUR CODE HERE '''
            # Fit least-squares model
            # Compute training error
            # Compute test error
            yhat = model.predict(Xtest)
            testError = np.sum((yhat - ytest)**2) / t
            print ("Test error = ", testError)
            # Plot model
            plt.figure()
            plt.plot(X,y,'b.', label = "Training data")
            plt.title('Training Data. p = {}'.format(p))
            
            # Choose points to evaluate the function
            Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
            
            yhat = model.predict(Xhat)
            plt.plot(Xhat,yhat,'g', label = "Least squares fit")


            '''YOUR CODE HERE'''
            #Predict on Xhat

            plt.legend()
            figname = os.path.join("..","figs","PolyBasis%d.pdf"%p)
            print("Saving", figname)
            plt.savefig(figname)
        
        
        
    elif question == "3.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("basisData")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']
        minErr = np.inf

       
        # get the number of rows(n) and columns(d)
        (n,d) = X.shape
 
        perm = np.random.permutation(n)
       
        Xperm = X[perm]
        yperm = y[perm]        
        # Split training data into a training and a validation set
    
        
        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set


# find mean accuracy over all rounds
        for s in range(-15,16):
            sigma = 2 ** s
            mean = 0
           
            num_folds = 10
            subset_size = n//num_folds#subset size= 20
            for i in range(num_folds):
                xtest = Xperm[i*subset_size:(i+1)*subset_size]
                yval = yperm[i*subset_size:(i+1)*subset_size]
               
                xtrain = np.r_[Xperm[0:i*subset_size],Xperm[(i+1)*subset_size:n]]#Need to fix these
                ytrain = np.r_[yperm[0:i*subset_size],yperm[(i+1)*subset_size:n]]#Need to fix these

                
                # Train on the training set
                model = linear_model.LeastSquaresRBF(sigma)
                model.fit(xtrain,ytrain)

            # Compute the error on the validation set
                yhat = model.predict(xtest)
                validError = np.sum((yhat - yval)**2)/subset_size
                mean = mean+validError
            mean=mean/10
            print("mean : %s" % mean)

            print("Error with sigma = {:e} = {}".format( sigma ,validError))

            # Keep track of the lowest validation error
            if mean < minErr:
                minErr = mean
                bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error = {:e}".format(bestSigma))

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        # Compute training error
        yhat = model.predict(X)
        trainError = np.sum((yhat - y)**2)/n
        print("Training error = {}".format(trainError))

        # Finally, report the error on the test set
        t = Xtest.shape[0]
        yhat = model.predict(Xtest)
        testError = np.sum((yhat - ytest)**2)/t
        print("Test error = {}".format(testError))

        # Plot model
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title('Training Data')

        Xhat = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xhat)
        plt.plot(Xhat,yhat,'g',label = "Least Squares with RBF kernel and $\sigma={}$".format(bestSigma))
        plt.ylim([-300,400])
        plt.legend()
        figname = os.path.join("..","figs","least_squares_rbf.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "4.1":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']
        
        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Fit least-squares estimator
        model = linear_model.WeightedLeastSquares()
       
        model.fit(X,y,1)
        print(model.w)

        ''' YOUR CODE HERE '''
        # Fit weighted least-squares estimator

        # Draw model prediction
        Xsample = np.linspace(np.min(X),np.max(X),1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample,yhat,'g-', label = "Least squares fit")
        plt.legend()
        figname = os.path.join("..","figs","least_squares_outliers.pdf")
        print("Saving", figname)
        plt.savefig(figname)


    elif question == "4.3":
        # loads the data in the form of dictionary
        data = utils.load_dataset("outliersData")
        X = data['X']
        y = data['y']

        # Fit least-squares estimator
        model = linear_model.LinearModelGradient()
        model.fit(X,y)
        print(model.w)

        # Plot data
        plt.figure()
        plt.plot(X,y,'b.',label = "Training data")
        plt.title("Training data")

        # Draw model prediction
        Xsample = np.linspace(np.min(X), np.max(X), 1000)[:,None]
        yhat = model.predict(Xsample)
        plt.plot(Xsample, yhat, 'g-', label = "Least squares fit")
        plt.legend()
        figname = os.path.join("..","figs","gradient_descent_model.pdf")
        print("Saving", figname)
        plt.savefig(figname)
