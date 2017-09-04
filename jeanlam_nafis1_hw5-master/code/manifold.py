from __future__ import division
import numpy as np
from numpy.linalg import norm
import pylab as plt
import utils
from pca import PCA
from pca import AlternativePCA
from utils import find_min

class MDS:

    def __init__(self, n_components):
        self.k = n_components

    def compress(self, X):
        n = X.shape[0]
        k = self.k

        # Compute Euclidean distances
        D = utils.euclidean_dist_squared(X,X)
        D = np.sqrt(D)

        # Initialize low-dimensional representation with PCA
        Z = PCA(k).fit(X).compress(X)

        # Solve for the minimizer
        z = find_min(self._fun_obj_z, Z.flatten(), 500, False, D)
        Z = z.reshape(n, k)
        return Z

    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()
# q3.1
class ISOMAP:
    def __init__(self, n_components, n_neighbours):
        self.numNeighbours = n_neighbours
        self.k = n_components

    def compress(self, X, n_components, n_neighbours):
        n = X.shape[0]
        k = self.k
        numNeighbours = self.numNeighbours
        
        # find the distances to every other point
        euclD = utils.euclidean_dist_squared(X,X)
        euclD = np.sqrt(euclD)

        knnD = np.zeros((n, n))

        # get the KNN of point i
        for i in range(n):
            # finds numNeighbours smallest distances from obj_i
            # +1 because it will always select itself as (distance of 0), and distances are non-negative
            minIndexes = np.argsort(euclD[i])[:numNeighbours+1]
            
            for index in minIndexes:
                # add distances of KNN_i to the distance matrix
                knnD[i, index] = euclD[i, index]
        
        D = np.zeros((n, n))        
        # get distance of every other path using only KNN
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = utils.dijkstra(knnD, i, j)

        Z = AlternativePCA(k).fit(X).compress(X)
        z = find_min(self._fun_obj_z, Z.flatten(), 500, False, D)
        Z = z.reshape(n, k)
        return Z
    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()

# q3.2
class ISOMAP_h:
    def __init__(self, n_components, n_neighbours):
        self.numNeighbours = n_neighbours
        self.k = n_components

    def compress(self, X, n_components, n_neighbours):
        n = X.shape[0]
        k = self.k
        numNeighbours = self.numNeighbours
        
        # find the distances to every other point
        euclD = utils.euclidean_dist_squared(X,X)
        euclD = np.sqrt(euclD)

        knnD = np.zeros((n, n))

        # get the KNN of point i
        for i in range(n):
            # finds numNeighbours smallest distances from obj_i
            # +1 because it will always select itself as (distance of 0), and distances are non-negative
            minIndexes = np.argsort(euclD[i])[:numNeighbours+1]
            
            for index in minIndexes:
                # add distances of KNN_i to the distance matrix
                knnD[i, index] = euclD[i, index]
        
        D = np.zeros((n, n))        
        # get distance of every other path using only KNN
        for i in range(n):
            for j in range(n):
                if i != j:
                    D[i, j] = utils.dijkstra(knnD, i, j)

        # get rid of infinite distances, and set it to max
        maxDist = 0
        for i in range(n):
            for j in range(n):
                if D[i, j] > maxDist and D[i, j] != np.inf:
                    maxDist = D[i,j]
        heuristicD = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if D[i, j] != np.inf:
                    heuristicD[i,j] = D[i, j]
                else:
                    print "Change", D[i,j], "to ", maxDist
                    heuristicD[i,j] = maxDist


        Z = AlternativePCA(k).fit(X).compress(X)
        z = find_min(self._fun_obj_z, Z.flatten(), 500, False, heuristicD)
        Z = z.reshape(n, k)
        return Z
    def _fun_obj_z(self, z, D):
        n = D.shape[0]
        k = self.k
        Z = z.reshape(n,k)

        f = 0.0
        g = np.zeros((n,k))
        for i in range(n):
            for j in range(i+1,n):
                # Objective Function
                Dz = norm(Z[i]-Z[j])
                s = D[i,j] - Dz
                f = f + (0.5)*(s**2)

                # Gradient
                df = s
                dgi = (Z[i]-Z[j])/Dz
                dgj = (Z[j]-Z[i])/Dz
                g[i] = g[i] - df*dgi
                g[j] = g[j] - df*dgj

        return f, g.flatten()
