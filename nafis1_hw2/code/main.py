import sys
import argparse
import pylab as plt
import numpy as np
import cnn
import utils
import knn
import decision_tree
import random_tree
import decision_forest
import kmeans
import dbscan
import kmedian
import quantize_image


if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1.1', '1.2', '2.1', '2.2', '3.1', '3.2', '3.3', '4.1', '4.2', '4.3'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1.1':
        dataset = utils.load_dataset('citiesSmall')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']
        #model = knn.fit(X,y,3)
        #model = knn.fit(X,y,1)
        model = knn.fit(X,y,10)

        y_pred_tr = knn.predict(model, X)
        y_pred_te = knn.predict(model, Xtest)
        trerror =utils.classification_error(y_pred_tr,y)
        teerror = utils.classification_error(y_pred_te, ytest)
       
        print(trerror)
        print(teerror)
 

        utils.plot_2dclassifier(model, Xtest, ytest)
        

        # part 1: implement knn.predict
        # part 2: print training and test errors for k=1,3,10 (use utils.classification_error)
        # part 3: plot classification boundaries for k=1 (use utils.plot_2dclassifier)

    if question == '1.2':
        dataset = utils.load_dataset('citiesBig1')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']
        model = cnn.fit(X,y,1)
        y_pred_tr = cnn.predict(model, X)
        y_pred_te = cnn.predict(model, Xtest)
        trerror =utils.classification_error(y_pred_tr,y)
        teerror = utils.classification_error(y_pred_te, ytest)
        print(trerror)
        print(teerror)

        utils.plot_2dclassifier(model, X, y)


        # part 1: implement cnn.py
        # part 2: print training/test errors as well as number of examples for k=1
        # part 3: plot classification boundaries for k=1

    if question == '2.1':
        dataset = utils.load_dataset('vowel')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        # # part 1: plot decision_tree as depth varies from 1 to 15
        train_errors = np.zeros(15)
        test_errors = np.zeros(15)

        for i in range(1,16):
            model = decision_tree.fit(X, y, i)
            y_pred = decision_tree.predict(model, X)
            training_error = np.sum(y_pred != y) / float(X.shape[0])

            # print "Training error:", training_error, "at depth", i
            y_pred = decision_tree.predict(model, Xtest)
            test_error = np.sum(y_pred != ytest) / float(Xtest.shape[0])
            # print "Test error:", test_error, "at depth", i

            train_errors[i-1] = training_error
            test_errors[i-1] = test_error
        x_vals = np.arange(1, 16)
                
        plt.title("Tree depth vs. training and test error")
        plt.plot(x_vals, train_errors, label="Training error")
        plt.plot(x_vals, test_errors, label="Testing error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()

        fname = "../figs/q2_1_1.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)

        plt.show()

        # part 2: max depth = inf

        model = decision_tree.fit(X, y, np.inf)
        y_pred = decision_tree.predict(model, X)
        print np.sum(y_pred != y)
        training_error = np.sum(y_pred != y) / float(X.shape[0])

        # print "Training error:", training_error, "at depth", i
        y_pred = decision_tree.predict(model, Xtest)
        test_error = np.sum(y_pred != ytest) / float(Xtest.shape[0])
        # print "Test error:", test_error, "at depth", i

        print("decision tree inf training error", training_error)
        print("decision tree inf testing error", test_error)

        # part 3: implement random_stump and report performance on random_tree
        for i in range(1,16):
            model = random_tree.fit(X, y, i)
            y_pred = random_tree.predict(model, X)
            training_error = np.sum(y_pred != y) / float(X.shape[0])

            # print "Training error:", training_error, "at depth", i
            y_pred = random_tree.predict(model, Xtest)
            test_error = np.sum(y_pred != ytest) / float(Xtest.shape[0])
            # print "Test error:", test_error, "at depth", i

            train_errors[i-1] = training_error
            test_errors[i-1] = test_error
            
        x_vals = np.arange(1,16)
        plt.title("Random Tree depth vs. training and test error")
        plt.plot(x_vals, train_errors, label="Training error")
        plt.plot(x_vals, test_errors, label="Testing error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()

        fname = "../figs/q2_1_3.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)

        plt.show()
        
        # part 4: bootstrap at every depth
        for i in range(1,16):
            # obtain a set of size X.shape(0) with replacement
            randomSelections = np.random.choice(X.shape[0], X.shape[0])
            X_bootstrap = X[randomSelections]
            y_bootstrap = y[randomSelections]
            
            # evaluate training error on original training data
            model = decision_tree.fit(X_bootstrap, y_bootstrap, i)
            y_pred = decision_tree.predict(model, X_bootstrap)
            training_error = np.sum(y_pred != y_bootstrap) / float(X.shape[0])

            # print "Training error:", training_error, "at depth", i
            y_pred = decision_tree.predict(model, Xtest)
            test_error = np.sum(y_pred != ytest) / float(Xtest.shape[0])
            # print "Test error:", test_error, "at depth", i

            train_errors[i-1] = training_error
            test_errors[i-1] = test_error
            
        x_vals = np.arange(1,16)
        plt.title("Decision Tree depth with bagging vs. training and test error")
        plt.plot(x_vals, train_errors, label="Training error")
        plt.plot(x_vals, test_errors, label="Testing error")
        plt.xlabel("Depth")
        plt.ylabel("Error")
        plt.legend()

        fname = "../figs/q2_1_4.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        plt.show()

    if question == '2.2':
        print "Question 2.2"
        dataset = utils.load_dataset('vowel')
        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']
       
        model = decision_forest.fit(X,y,max_depth=np.inf, n_bootstrap=50)
        y_pred_tr = decision_forest.predict(model, X)
        y_pred_te = decision_forest.predict(model, Xtest)
        trerror =utils.classification_error(y_pred_tr,y)
        teerror = utils.classification_error(y_pred_te, ytest)
        print(trerror)
        print(teerror)

        # part 1: decision trees
        model = decision_forest.fit(X, y)
        y_pred = decision_forest.predict(model, X)
        training_error = utils.classification_error(y_pred, y)

        y_pred = decision_forest.predict(model, Xtest)
        test_error = utils.classification_error(y_pred, ytest)
        # model = decision_forest.fit(X,y,max_depth=np.inf, n_bootstrap=50)
        # y_pred_tr = decision_forest.predict(model, X)
        # y_pred_te = decision_forest.predict(model, Xtest)
        # trerror =utils.classification_error(y_pred_tr,y)
        # teerror = utils.classification_error(y_pred_te, ytest)
        # print("Training error", trerror)
        # print("Testing error", teerror)
        print "1) Training error: ", training_error
        print "1) Test error: ", test_error

        # # part 2: bootstrap sampling
        # model = decision_forest.fit(X, y)
        # y_pred = decision_forest.predict(model, X)
        # training_error = utils.classification_error(y_pred, y)

        # y_pred = decision_forest.predict(model, Xtest)
        # test_error = utils.classification_error(y_pred, ytest)

        # print "2) Training error: ", training_error
        # print "2) Test error: ", test_error

        # # part 3: random trees
        # model = decision_forest.fit(X, y)
        # y_pred = decision_forest.predict(model, X)
        # training_error = utils.classification_error(y_pred, y)

        # y_pred = decision_forest.predict(model, Xtest)
        # test_error = utils.classification_error(y_pred, ytest)

        # print "3) Training error: ", training_error
        # print "3) Test error: ", test_error
        # # part 4: random trees + bootstrap sampling
        # model = decision_forest.fit(X, y)
        # y_pred = decision_forest.predict(model, X)
        # training_error = utils.classification_error(y_pred, y)

        # y_pred = decision_forest.predict(model, Xtest)
        # test_error = utils.classification_error(y_pred, ytest)

        # print "4) Training error: ", training_error
        # print "4) Test error: ", test_error

    if question == '3.1':
        X = utils.load_dataset('clusterData')['X']

        model = kmeans.fit(X, k=4)
        utils.plot_2dclustering(X, model['predict'](model, X))
        print("Displaying figure...")
        plt.show()

        # part 1: implement kmeans.error
        error = model['error'](model,X)
        print "1) error", error

        # part 2: get clustering with lowest error out of 50 random initialization
        min_error = 100000000
        min_model = None
        for i in range(0,50):
            model = kmeans.fit(X, k=4)
            error = model['error'](model,X)
            if error < min_error:
                utils.plot_2dclustering(X, model['predict'](model, X))
                min_model = model
                min_error = error
        utils.plot_2dclustering(X, min_model['predict'](min_model, X))
        title = "Cluster data with minimum error of " + str(min_error)
        plt.title(title)
        
        fname = "../figs/q3_1_2.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        print min_error
        plt.show()

    if question == '3.2':
        X = utils.load_dataset('clusterData')['X']

        # part 3: plot min error across 50 random inits, as k is varied from 1 to 10
        min_error = 100000000
        min_model = None
        errorArr = np.zeros(10)
        for j in range(1,11):
            for i in range(0,50):
                model = kmeans.fit(X, k=j)
                error = model['error'](model,X)
                if error < min_error:
                    # utils.plot_2dclustering(X, model['predict'](model, X))
                    min_model = model
                    min_error = error
            
            errorArr[j-1] = min_error
            min_error = 100000000
        
        x_val = np.arange(1,11)
        
        plt.title("Minimum Error vs. k-Value")
        plt.plot(x_val, errorArr, label="Minimum error observed", linestyle='--', marker='o')
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.xticks(np.arange(min(x_val), max(x_val)+1, 1.0))
        plt.legend()
        
        fname = "../figs/q3_2_3.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        
        plt.show()
        print errorArr
        for item in range(0, len(errorArr)-1):
            print "k=", item, "- k=", item+1, " == ", errorArr[item] - errorArr[item+1]

    if question == '3.3':
        X = utils.load_dataset('clusterData2')['X']

        # part 1: using clusterData2, plot min error across 50 random inits, as k is varied from 1 to 10
        min_error = 100000000
        min_model = None
        min_error_k = 0
        for i in range(0,50):
            model = kmeans.fit(X, k=4)
            error = model['error'](model,X)
            if error < min_error:
                utils.plot_2dclustering(X, model['predict'](model, X))
                min_model = model
                min_error = error
        utils.plot_2dclustering(X, min_model['predict'](min_model, X))
        title = "Cluster data with minimum error of " + str(min_error)
        plt.title(title)
        
        fname = "../figs/q3_3_1.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        print min_error
        plt.show()

        # part 2: determining the elbow
        min_error = 100000000
        min_model = None
        errorArr = np.zeros(10)
        for j in range(1,11):
            for i in range(0,50):
                model = kmeans.fit(X, k=j)
                error = model['error'](model,X)
                if error < min_error:
                    # utils.plot_2dclustering(X, model['predict'](model, X))
                    min_model = model
                    min_error = error
            
            errorArr[j-1] = min_error
            min_error = 100000000
        
        x_val = np.arange(1,11)
        
        plt.title("Minimum Error vs. k-Value")
        plt.plot(x_val, errorArr, label="Minimum error observed", linestyle='--', marker='o')
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.xticks(np.arange(min(x_val), max(x_val)+1, 1.0))
        plt.legend()
        
        fname = "../figs/q3_3_2.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        
        plt.show()
        print errorArr
        for item in range(0, len(errorArr)-1):
            print "k=", item, "- k=", item+1, " == ", errorArr[item] - errorArr[item+1]

        # part 3: implement kmedians.py
        model = kmedian.fit(X, k=4)
        error = model['error'](model,X)
        
        utils.plot_2dclustering(X, model['predict'](model, X))
        plt.title("Cluster data with K-median")
        
        fname = "../figs/q3_3_3.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        print error
        plt.show()
        
        # part 4: plot kmedians.error
        min_error = 100000000
        min_model = None
        errorArr = np.zeros(10)
        for j in range(1,11):
            for i in range(0,50):
                model = kmedian.fit(X, k=j)
                error = model['error'](model,X)
                if error < min_error:
                    # utils.plot_2dclustering(X, model['predict'](model, X))
                    min_model = model
                    min_error = error
            
            errorArr[j-1] = min_error
            min_error = 100000000
        
        x_val = np.arange(1,11)
        
        plt.title("Minimum Error vs. k-Value")
        plt.plot(x_val, errorArr, label="Minimum error observed", linestyle='--', marker='o')
        plt.xlabel("k")
        plt.ylabel("Error")
        plt.xticks(np.arange(min(x_val), max(x_val)+1, 1.0))
        plt.legend()
        fname = "../figs/q3_3_4.pdf"
        plt.savefig(fname)
        print("Figure saved as ", fname)
        # print error
        plt.show()

        for item in range(0, len(errorArr)-1):
            print "k=", item, "- k=", item+1, " == ", errorArr[item] - errorArr[item+1]

    if question == '4.1':
        img = utils.load_dataset('dog')['I']/255
        plt.imshow(img)
        print("Displaying figure...")
        #quantize_image.quantize_image(img,4)
        #plt.imshow(quantize_image.quantize_image(img,1))
        #plt.imshow(quantize_image.quantize_image(img,2))
        #plt.imshow(quantize_image.quantize_image(img,4))

        plt.imshow(quantize_image.quantize_image(img,6))
        plt.show()

        # part 1: implement quantize_image.py
        # part 2: use it on the doge

    if question == '4.2':
        X = utils.load_dataset('clusterData2')['X']
        model = dbscan.fit(X, radius2=1000, min_pts=3)
        y = model['predict'](model, X)
        utils.plot_2dclustering(X,y)
        print("Displaying figure...")
        plt.show()

    if question == '4.3':
        dataset = utils.load_dataset('animals')
        X = dataset['X']
        animals = dataset['animals']
        traits = dataset['traits']

        model = dbscan.fit(X, radius2=15, min_pts=3)
        y = model['predict'](model, X)

        for kk in range(max(y)+1):
            print('Cluster {}: {}'.format(kk+1, ' '.join(animals[y==kk])))
