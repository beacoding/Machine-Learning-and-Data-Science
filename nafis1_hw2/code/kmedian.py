import numpy as np
import pylab as plt
import utils

def fit(X, k, do_plot=False):
    N, D = X.shape
    y = np.ones(N)

    medians = np.zeros((k, D))
    # random initialization
    for kk in range(k):
        i = np.random.randint(N)
        medians[kk] = X[i]

    dist = np.zeros((N, k))
    while True:
        y_old = y

        # Compute L1 distance to each median
        for n in range(N):
            current_obj = X[n,]
            for i in range(k):                
                current_median = medians[i,]
                distance = np.abs(current_obj[0] - current_median[0]) + np.abs(current_obj[1] - current_median[1])
                # print distance
                dist[n,i] = distance

        dist[np.isnan(dist)] = np.inf
        y = np.argmin(dist, axis=1)

        # Update medians
        for kk in range(k):
            # medians[kk] = X[y==kk].median(axis=0)
            cluster = X[y==kk]
            median_x = np.median(cluster, axis=0)
            # print median_x
            medians[kk] = median_x

        changes = np.sum(y != y_old)
        print('Running K-medians, changes in cluster assignment = {}'.format(changes))

        # Stop if no point changed cluster
        if changes == 0:
            break

    model = dict()
    model['medians'] = medians
    model['predict'] = predict
    model['error'] = error

    if do_plot and D == 2:
        utils.plot_2dclustering(X, y)
        print("Displaying figure...")
        plt.show()

    return model

def predict(model, X):
    medians = model['medians']
    dist2 = utils.euclidean_dist_squared(X, medians)
    # print np.argmin(dist2, axis=1)
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
        # closest median predicted
        closest_median = model["medians"][predictions[n],]
        # current object
        current_x = X[n,]
        distance = (np.abs(closest_median[0] - current_x[0]) + np.abs(closest_median[1] - current_x[1]))
        # distance = 0
       
        # break;
        sum_error += distance
    # print model["medians"]

    return sum_error