#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 20:20:08 2019

@author: theodorelee
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1.,1.,3.],[2.,3.,2.],[1.,1.,1.],[1.,1.,2.],[2.,2.,3.],[3,3,2],[1,3,2],[4,3,5],[5,5,5]])
y = np.array([[4,4,-1.07846],[3,3,1.214359],[2,2,0.307180],[2,3,-0.385641],[2,1,-0.078461],[1,1,1.61436],[1,2,0.81436],[2,1,-0.06410],[1,2,1.54590]])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(np.squeeze(np.asarray(x[:,0])), np.squeeze(np.asarray(x[:,1])), np.squeeze(np.asarray(x[:,2])), c='b', cmap='Greens');
ax.scatter3D(np.squeeze(np.asarray(y[:,0])), np.squeeze(np.asarray(y[:,1])), np.squeeze(np.asarray(y[:,2])), c='r', cmap='Greens');

def cov2(a,b):
    matrix = []
    for i in range(3):
        for j in range(3):
            matrix.append( np.cov(a[:,i],b[:,j])[0,1])
            
    matrix = np.asarray(matrix)
    matrix= np.reshape(matrix, (3,3))
    return matrix
    
Sxx = np.cov(x, rowvar=False)
Syy = np.cov(y, rowvar=False)
Sxy = cov2(x,y)
Syx = cov2(y,x)

aproduct = np.dot(np.dot(np.dot(np.linalg.inv(Sxx),Sxy), np.linalg.inv(Syy)),Syx)
bproduct = np.dot(np.dot(np.dot(np.linalg.inv(Syy),Syx), np.linalg.inv(Sxx)),Sxy)

R = np.linalg.eig(aproduct)[0]
Aweights = np.linalg.eig(aproduct)[1]
Bweights = np.linalg.eig(bproduct)[1]


