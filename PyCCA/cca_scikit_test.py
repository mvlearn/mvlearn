#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 00:18:49 2019

@author: theodorelee
"""

from sklearn.cross_decomposition import CCA
import matplotlib.pyplot as plt
import numpy as np

X = np.array([[0., 0., 1.], [1.,0.,0.], [2.,2.,2.], [3.,5.,4.]])
Y = np.array([[0.1, -0.2], [0.9, 1.1], [6.2, 5.9], [11.9, 12.3]])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(np.squeeze(np.asarray(X[:,0])), np.squeeze(np.asarray(X[:,1])), np.squeeze(np.asarray(X[:,2])), c='b', cmap='Greens');
ax.scatter3D(np.squeeze(np.asarray(Y[:,0])), np.squeeze(np.asarray(Y[:,1])), c='r', cmap='Greens');


cca = CCA(n_components=2)
cca.fit(X, Y)
X_c, Y_c = cca.transform(X, Y)