from __future__ import division
import numpy as np

def fit(X, y):
    """ YOUR CODE HERE FOR Q4.3 """
    N, D = X.shape

    # Compute the number of class labels
    C = np.unique(y).size

    # Create a mapping from the labels to 0,1,2,...
    # so that we can store things in numpy arrays
    labels = dict()
    for index, label in enumerate(np.unique(y)):
        labels[index] = label

    # Compute the probability of each class i.e p(y==c)
    counts = np.zeros(C)
    
    for index, label in labels.items():
        counts[index] = np.sum(y==label)
        p_y = counts / N
    print p_y
    # Compute the conditional probabilities i.e.
    # p(x(i,j)=1 | y(i)==c) as p_xy
    # p(x(i,j)=0 | y(i)==c) as p_xy
    p_xy = np.zeros((D,C,2))
    # print p_xy.shape
    #py = p_y.copy()

    # count_x_0_y_0 = 0
    # count_x_1_y_0 = 0
    # count_x_1_y_1 = 0
    # count_x_0_y_1 = 0
    # for each feature
    # for d in range(D):
    #     # for each object 
    #     for c in range(C):
    #         for x in range(2):
    #             # find the number of times that X[,d] == x and y[]
    #             # feature column = X[,d]
    #             col = X[:,d]
    count = 0
    # print "There are", N, " objects"
    for d in range(D):
        for c in range(C):
            for x in range(2):
                for n in range(N):
                    if X[n,d] == x and y[n] == c+1:
                        count += 1
                # print "Total counts found satisfying checking if X[n,", d, "] == ", x ," and y == ", c+1, "== ", count, ". P_y == ", p_y[c]
                
                p_xy[d, c, x] = (count/N) / p_y[c]
                count = 0

    # print p_xy
    
  # #for n in range(N):
  #   for c in range(C): 
  #       for d in range(D):
  #           if X[d,c] == 1:
  #                count_x_1 +=1

  #           elif X[d,c] == 0:
  #                count_x_0 +=1
  #           p_xy[d,c,0] = count_x_0
  #           p_xy[d,c,1] = count_x_1
    # Save parameters in model as dict
    model = dict()

    model["p_y"] = p_y
    model["p_xy"] = p_xy
    model["n_classes"] = C
    model["labels"] = labels

    return model


def fit_wrong(X, y):
    N, D = X.shape

    # Compute the number of class labels
    C = np.unique(y).size

    # Create a mapping from the labels to 0,1,2,...
    # so that we can store things in numpy arrays
    labels = dict()
    for index, label in enumerate(np.unique(y)):
        labels[index] = label

    # Compute the probability of each class i.e p(y==c)
    counts = np.zeros(C)

    for index, label in labels.items():
        counts[index] = np.sum(y==label)
        p_y = counts / N

    # Compute the conditional probabilities i.e.
    # p(x(i,j)=1 | y(i)==c) as p_xy
    # p(x(i,j)=0 | y(i)==c) as p_xy
    p_xy = 0.5 * np.ones((D, C, 2))

    # Save parameters in model as dict
    model = dict()

    model["p_y"] = p_y
    model["p_xy"] = p_xy
    model["n_classes"] = C
    model["labels"] = labels

    return model

def predict(model, X):
    N, D = X.shape
    C = model["n_classes"]
    p_xy = model["p_xy"]
    p_y = model["p_y"]
    labels = model["labels"]

    y_pred = np.zeros(N)

    for n in range(N):
        # Compute the probability for each class
        # This could be vectorized but we've tried to provide
        # an intuitive version.
        probs = p_y.copy()

        for d in range(D):
            if X[n, d] == 1:
                for c in range(C):
                    probs[c] *= p_xy[d, c, 1]

            elif X[n, d] == 0:
                for c in range(C):
                    probs[c] *= p_xy[d, c, 0]

        y_pred[n] = labels[np.argmax(probs)]

    return y_pred
