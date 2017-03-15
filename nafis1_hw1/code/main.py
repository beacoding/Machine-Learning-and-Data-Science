import sys
import argparse

import utils
import pylab as plt

import numpy as np
import naive_bayes
import decision_stump
import decision_tree
import mode_predictor

import math

from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True, 
        choices=["1.1", "1.2", "2.1", "2.2", "3.1", "4.3"]) 

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1.1":
        # Q1.1 - This should print the answers to Q 1.1

        # Load the fluTrends dataset
        X, names = utils.load_dataset("fluTrends")

        """ YOUR CODE HERE"""
        # part 1: min, max, mean, median and mode

        # N = Number of objects
        # D = Number of features
        N, D = X.shape        

        min_val = 100
        max_val = 0
        sum_val = 0
        median_val = 0
        list_vals = X.ravel()
        list_vals.sort()

        if len(list_vals) % 2 == 0:
            median_val = (list_vals[(len(list_vals)/2)] + list_vals[(len(list_vals)/2)-1])/2
        else:
            median_val = list_vals[(len(list_vals) - 1) / 2]

        for feature in range(D):       
            
            mode_val = utils.mode(X[:][:])

            for obj in range(N):
                if X[obj][feature] > max_val:
                    max_val = X[obj][feature]
                if X[obj][feature] < min_val:
                    min_val = X[obj][feature]
                sum_val = sum_val + X[obj][feature]          

        print "Minimum value is ", min_val
        print "Maximum value is ", max_val
        print "Mean value is ", (sum_val / (N*D))
        print "Median value is ", median_val
        print "Mode value is ", mode_val

        # part 2: quantiles
        # TODO: do we treat it the same as median, and average between two values? Or just take the ceil?
        length = len(list_vals) - 1

        quartile_10 = np.percentile(list_vals, 10)
        quartile_25 = np.percentile(list_vals, 25)

        # should be same as median_val
        quartile_50 = np.percentile(list_vals, 50)

        quartile_75 = np.percentile(list_vals, 75)
        quartile_90 = np.percentile(list_vals, 90)



        print "10 quantile ", quartile_10
        print "25 quantile ", quartile_25
        print "50 quantile ", quartile_50
        print "75 quantile ", quartile_75
        print "90 quantile ", quartile_90


        # part 3: maxMean, minMean, maxVar, minVar
        max_mean = 0
        max_mean_region = ""
        min_mean = 100
        min_mean_region = ""
        max_var = 0
        max_var_region = ""
        min_var = 100
        min_var_region = ""

        for feature in range(D):                   
            sum_val = 0
            for obj in range(N):
                sum_val = sum_val + X[obj][feature]          
            
            mean = sum_val / N
            
            if mean > max_mean:
                max_mean = mean
                max_mean_region = names[feature]
            elif mean < min_mean:
                min_mean = mean
                min_mean_region = names[feature]

            # compute the variance -- squared differences from the mean
            sum_diff = 0
            for obj in range(N):
                sum_diff = sum_diff + (X[obj][feature] - mean)**2
            variance = sum_diff / N

            if variance > max_var:
                max_var = variance
                max_var_region = names[feature]
            elif variance < min_var:
                min_var = variance
                min_var_region = names[feature]


        print "Region ", max_mean_region, "has max mean of ", max_mean
        print "Region ", min_mean_region, "has min mean of ", min_mean
        print "Region ", max_var_region, "has max var of ", max_var
        print "Region ", min_var_region, "has min var of ", min_var


        # part 4: correlation between columns
        max_correlation = 0
        min_correlation = 100
        max_correlation_region_1 = ""
        max_correlation_region_2 = ""
        min_correlation_region_1 = ""
        min_correlation_region_2 = ""
        
        matrix = np.corrcoef(X.T)
        for arr in range(D):
            for arr2 in range(D):
                # one means itself
                value = matrix[arr, arr2]
                if str(value) != "1.0":
                    print value
                    if value > max_correlation and value < 1.0:
                        max_correlation = value
                        max_correlation_region_1 = names[arr]
                        max_correlation_region_2 = names[arr2]
                    elif value < min_correlation:
                        min_correlation = value
                        min_correlation_region_1 = names[arr]
                        min_correlation_region_2 = names[arr2]
        # for feature_1 in range(D):
        #     for feature_2 in range(feature_1,D):
        #         # print "Comparing ", names[feature_1], "and", names[feature_2]
        #         if feature_1 != feature_2:

                    # sum_x = 0
                    # sum_y = 0
                    # sum_x_squared = 0
                    # sum_y_squared = 0
                    # sum_xy = 0
                    
                    # for obj in range(N):                            
                    #     sum_x = sum_x + X[obj][feature_1]
                    #     sum_y = sum_y + X[obj][feature_2]
                    #     sum_x_squared = sum_x_squared + (X[obj][feature_1]**2)
                    #     sum_y_squared = sum_y_squared + (X[obj][feature_2]**2)
                    #     sum_xy = sum_xy + (X[obj][feature_1] * X[obj][feature_2])

                    # numerator = (N * sum_xy) - (sum_x * sum_y)
                    # denominator = (((N * sum_x_squared) - sum_x**2) * ((N * sum_y_squared) - sum_y**2))**0.5
                    # correlation = numerator / denominator

                    # if correlation > max_correlation:
                    #     max_correlation = correlation
                    #     max_correlation_region_1 = names[feature_1]
                    #     max_correlation_region_2 = names[feature_2]
                    # elif correlation < min_correlation:
                    #     min_correlation = correlation
                    #     min_correlation_region_1 = names[feature_1]
                    #     min_correlation_region_2 = names[feature_2]
                   
        print "Highest correlation between", max_correlation_region_1, "region and", max_correlation_region_2, "region with correlation value", max_correlation
        print "Lowest correlation between", min_correlation_region_1, "region and", min_correlation_region_2, "region correlation value", min_correlation
        
    elif question == "1.2":
        # Q1.2 - This should plot the answers to Q 1.2
        # Load the fluTrends dataset
        X, names = utils.load_dataset("fluTrends")

        """ YOUR CODE HERE"""
        # Q 1.2.1
        N, D = X.shape


        # 1) Plot 
        plt.figure(1)
        ax = plt.subplot(111)
        ax.set_xlim(1, N)
        plt.xlabel("Week")
        plt.ylabel("Percentage")
        
        for index in range(D):
            plt.plot(X[:,index], '-o', label=names[index], color=np.random.rand(3,1))
        plt.legend(loc='upper right')
        plt.suptitle('Percentage of Flu Illness over 52 Weeks in 2005-2006')
        
        fname = "../figs/q1.2_1.png"
        plt.savefig(fname)
        plt.close()

        # 2) BOXPLOT grouping data by weeks
        # prepare data
        data = []
        for index in range(N):
            data.append(X[index:,])

        plt.figure(2, figsize=(15,10))
        plt.boxplot(data)
        
        plt.suptitle('Distribution of Flu Illness over 52 Weeks in 2005-2006')
        plt.xlabel("Week")
        plt.ylabel("Percentage")
        # plt.show()
        fname = "../figs/q1.2_2.png"
        plt.savefig(fname)
        plt.close()

        # 3) single histogram showing distribution of each value in X
        # prepare data
        allValues = []
        for xIndex in range(N):
            for dIndex in range(D):
                allValues.append(X[xIndex,dIndex])
        # create a new figure, and plot
        plt.figure(3)
        plt.hist(allValues)
        plt.suptitle('Distribution of Flu Illness over 52 Weeks in 2005-2006')
        plt.xlabel("Bin")
        plt.ylabel("Frequency")
        fname = "../figs/q1.2_3.png"
        plt.savefig(fname)
        plt.close()
    
        # 4) TODO: single histogram showing distribution of each column in X
        plt.figure(4, figsize=(15,10))
        # # prepare data group by column

        colDataNames = []
        for index in range(D):
            colDataNames.append(names[index])
        plt.xlabel("Bin Value")
        plt.ylabel("Frequency")
        

        plt.hist(X, histtype="bar", label=colDataNames)
        plt.suptitle('Distribution of Flu Illness per Region over 52 Weeks in 2005-2006')
        plt.legend()
        fname = "../figs/q1.2_4.png"
        plt.savefig(fname)
        plt.close()
    
        # 5) scatterplot lowest correlation = NE, Mtn
        
        plt.figure(5)
        plt.suptitle('Percentage of Flu in NE vs. Mtn')
        plt.scatter(X[:,0],X[:,7])
        plt.xlabel("Percentage of Flu in NE")
        plt.ylabel("Percentage of Flu in Mtn")
        fname = "../figs/q1.2_5.png"
        plt.savefig(fname)        
        

        # 6) scatterplot highest correlation = MidAtl, ENCentral
        plt.figure(6)
        plt.suptitle('Percentage of Flu in MidAtl vs. ENCentral')
        plt.scatter(X[:,1],X[:,2])
        plt.xlabel("Percentage of Flu in MidAtl")
        plt.ylabel("Percentage of Flu in ENCentral")
        fname = "../figs/q1.2_6.png"
        plt.savefig(fname)        
        
    elif question == "2.1":
        # Q2.1 - Decision Stump with the inequality rule Implementation

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")

        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate majority predictor model
        model = mode_predictor.fit(X, y)
        y_pred = mode_predictor.predict(model, X)

        error = np.sum(y_pred != y) / float(X.shape[0])
        print("Mode predictor error: %.3f" % error)

        # 3. Evaluate decision stump with equality rule
        model = decision_stump.fit_equality(X, y)
        y_pred = decision_stump.predict_equality(model, X)

        error = np.sum(y_pred != y) / float(X.shape[0])
        print("Decision Stump with equality rule error: %.3f" 
              % error)

        # 4. Evaluate decision stump with inequality rule

        """ YOUR CODE HERE"""
        model = decision_stump.fit(X, y)
        y_pred = decision_stump.predict(model, X)

        error = np.sum(y_pred != y) / float(X.shape[0])
        print("Decision Stump with inequality rule error: %.3f" 
              % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)
        fname = "../figs/q2.1_decisionBoundary.pdf"
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        

    elif question == "2.2":
        # Q2.2 - Decision Tree with depth 2

        # 1. Load citiesSmall dataset
        dataset = utils.load_dataset("citiesSmall")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree 
        # model = decision_tree.fit(X, y, maxDepth=2)
        # # print model
        # y_pred = decision_tree.predict(model, X)
        # error = np.mean(y_pred != y)

        # print("Error: %.3f" % error)

        # # 3. Evaluate decision tree that uses information gain
        # tree = DecisionTreeClassifier(max_depth=3)
        # tree.fit(X, y)

        # y_pred = tree.predict(X)
        # error = np.mean(y_pred != y)

        # print("Error: %.3f" % error)
      
        for maxDepth in range(2,15):
            print "******* MAX DEPTH =", maxDepth, "***********"
            # 2. Evaluate decision tree 
            model = decision_tree.fit(X, y, maxDepth=maxDepth)
            # print model
            y_pred = decision_tree.predict(model, X)
            error = np.mean(y_pred != y)
            # print model
            print("Error: %.3f" % error)

            # 3. Evaluate decision tree that uses information gain
            tree = DecisionTreeClassifier(max_depth=maxDepth+1)
            tree.fit(X, y)

            y_pred = tree.predict(X)
            error = np.mean(y_pred != y)

            print("Error: %.3f" % error)

    elif question == "3.1":
        dataset = utils.load_dataset("citiesSmall")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        k = []
        l = []
        
        
        for n in range(1,16):
            
            model = DecisionTreeClassifier(criterion='entropy', max_depth=n)
            model.fit(X, y)

            #print("depth: %.3f" % n)        

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)
            k.append(tr_error)
            #print("training error: %.3f" % tr_error)        

           # result_array = np.append(result_array, [result], axis=0)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            l.append(te_error)
           # print("test error: %.3f" % te_error)        

    
        
            
        plt.plot(k)
        plt.plot(l)
        #plt.ylabel()
        plt.show()

    elif question == "4.3":
        # Q4.3 - Train Naive Bayes

        # 1. Load dataset
        dataset = utils.load_dataset("newsgroups")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        
        # 2. Evaluate the decision tree model with depth 20
        model = DecisionTreeClassifier(criterion='entropy', max_depth=20)
        model.fit(X, y)
        y_pred = model.predict(X_valid)

        v_error = np.mean(y_pred != y_valid)
        print("Decision Tree Validation error: %.3f" % v_error)        
  
        # 3. Evaluate the Naive Bayes Model
        model = naive_bayes.fit(X, y)

        y_pred = naive_bayes.predict(model, X_valid)

        v_error = np.mean(y_pred != y_valid)
        print("Naive Bayes Validation error: %.3f" % v_error)

