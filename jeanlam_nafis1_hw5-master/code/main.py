from __future__ import division
import sys
import argparse
import pylab as plt
import numpy as np
from numpy.linalg import norm

import utils
from pca import PCA, AlternativePCA, RobustPCA
from manifold import MDS, ISOMAP, ISOMAP_h

if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument('-q','--question', required=True,
        choices=['1.2', '2.1', '3', '3.1', '3.2'])

    io_args = parser.parse_args()
    question = io_args.question

    if question == '1.2':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape
        k=5;
        X = utils.standardize_cols(X)        # standardize columns

        model = PCA(k=2);
        model.fit(X)
        Z = model.compress(X);
            

        # Plot the matrix
        plt.imshow(Z)
        utils.savefig('q1_unsatisfying_visualization_1.png')

    ## Randomly plot two features, and label all points
        
        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0],Z[i,1]))
        utils.savefig('q1_unsatisfying_visualization_2.png')
        v=1-norm(np.dot(Z,model.W)-X,'fro')**2/ norm(X,'fro')**2
        print v #The variance


    if question == '2.1':
        X = utils.load_dataset('highway')['X'].astype(float)/255
        n,d = X.shape
        h,w = 64,64 # height and width of each image

        # the two variables below are parameters for the foreground/background extraction method
        # you should just leave these two as default.

        k = 5 # number of PCs
        threshold = 0.04 # a threshold for separating foreground from background

        model = RobustPCA(k=k)
        model.fit(X)
        Z = model.compress(X)
        Xhat = model.expand(Z)

        # save 10 frames for illustration purposes
        for i in range(10):
            plt.subplot(1,3,1)
            plt.title('Original')
            plt.imshow(X[i].reshape(h,w).T, cmap='gray')
            plt.subplot(1,3,2)
            plt.title('Reconstructed')
            plt.imshow(Xhat[i].reshape(h,w).T, cmap='gray')
            plt.subplot(1,3,3)
            plt.title('Thresholded Difference')
            plt.imshow(1.0*(abs(X[i] - Xhat[i])<threshold).reshape(h,w).T, cmap='gray')
            utils.savefig('q2_highway_{:03d}.jpg'.format(i))

    if question == '3':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        model = MDS(n_components=2)
        Z = model.compress(X)

        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('MDS Features')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))
        utils.savefig('q3_MDS_animals.png')

    if question == '3.1':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        model = ISOMAP(n_components=2, n_neighbours=3)
        Z = model.compress(X, n_components=2, n_neighbours=3)

        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('ISOMAP Features')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))
        utils.savefig('q3_ISOMAP_animals.png')

    if question == '3.2':
        dataset = utils.load_dataset('animals')
        X = dataset['X'].astype(float)
        animals = dataset['animals']
        n,d = X.shape

        model = ISOMAP_h(n_components=2, n_neighbours=2)
        Z = model.compress(X, n_components=2, n_neighbours=2)

        fig, ax = plt.subplots()
        ax.scatter(Z[:,0], Z[:,1])
        plt.ylabel('z2')
        plt.xlabel('z1')
        plt.title('ISOMAP_h Features')
        for i in range(n):
            ax.annotate(animals[i], (Z[i,0], Z[i,1]))
        utils.savefig('q3_ISOMAP_heuristic_animals.png')
