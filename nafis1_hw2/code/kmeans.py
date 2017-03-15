import numpy as np
import pylab as plt
import utils

def fit(X, k, do_plot=False):
    N, D = X.shape
    y = np.ones(N)

    means = np.zeros((k, D))
    for kk in range(k):
        i = np.random.randint(N)
        means[kk] = X[i]

    while True:
        y_old = y

        # Compute euclidean distance to each mean
        dist2 = utils.euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        y = np.argmin(dist2, axis=1)

        # Update means
        for kk in range(k):
            means[kk] = X[y==kk].mean(axis=0)

        changes = np.sum(y != y_old)
        print('Running K-means, changes in cluster assignment = {}'.format(changes))

        # Stop if no point changed cluster
        if changes == 0:
            break

    model = dict()
    model['means'] = means
    model['predict'] = predict
    model['error'] = error

    if do_plot and D == 2:
        utils.plot_2dclustering(X, y)
        print("Displaying figure...")
        plt.show()

    return model

def predict(model, X):
    means = model['means']
    dist2 = utils.euclidean_dist_squared(X, means)

    # print np.argmin(dist2, axis=1)

    dist2[np.isnan(dist2)] = np.inf

    return np.argmin(dist2, axis=1)

def error(model, X):
    """ YOUR CODE HERE """
    N, D = X.shape
    sum_error = 0
    lowest_dist = 100000000
    predictions = model["predict"](model, X)
    # print predictions.shape
    for n in range(0,N):
        # for d in range(0,D):
        # closest mean predicted
        closest_mean = model["means"][predictions[n],]
        # current object
        current_x = X[n,]
        distance = np.sqrt((np.abs(closest_mean[0] - current_x[0])**2 + np.abs(closest_mean[1] - current_x[1])**2))
        # distance = 0
       
        # break;
        sum_error += distance
    # print model["means"]

    return sum_error