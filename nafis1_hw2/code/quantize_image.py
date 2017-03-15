
import sys
import argparse
import pylab as plt
import numpy as np
import cnn
import utils
import kmeans

def quantize_image(I,b):

#returns Q-img matrix :2d matrix

   # k=2**b# number of clusters

    [height,width,ncolors] = I.shape
    z = np.multiply(height,width)
    
    X = np.reshape(I,(height*width,3))

    model = kmeans.fit(X,2**b)
    #use predict 
    y = model['predict'](model,X);
    #print (y)
    Xhat = np.zeros((height*width ,3));
    #print (Xhat)
    for i in range(np.multiply(height,width)):
        Xhat[i,:] = model['means'][y[i],:]


    # dequantize
    DQ = np.reshape(Xhat,[height,width,3]);
   # print(DQ)
    return DQ

