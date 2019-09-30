from numpy import *
import numpy as np

def kernel_icd(X, kernel, m = None, precision = 1e-6):
    """Approximates a kernel matrix using incomplete Cholesky decomposition (ICD).

    Input:	- X: data matrix in row format (each data point is a row)
                - kernel: the kernel function. It should calculate on the diagonal!
                - kpar: vector containing the kernel parameters.
                - m: maximal rank of solution
                - precision: accuracy parameter of the ICD method
    Output:	- G: "narrow tall" matrix of the decomposition K ~= GG'
                - subset: indices of data selected for low-rank approximation

    USAGE: G = km_kernel_icd(X,ktype,kpar,m,precision)

    Based on code from Steven Van Vaerenbergh (steven *at* gtas.dicom.unican.es), 2010.

    The algorithm in this file is based on the following publication:
    Francis R. Bach, Michael I. Jordan. "Kernel Independent Component
    Analysis", Journal of Machine Learning Research, 3, 1-48, 2002.

    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, version 3 (as included and available at
    http://www.gnu.org/licenses).
    """
    n = X.shape[0]
    if m is None:
        m = n

    perm = arange(n)  #permutation vector
    d = zeros((1, n))  #diagonal of the residual kernel matrix
    G = zeros((n, m))
    subset = zeros((1, m))

    for i in range(m):
        x = X[perm[i : n+1], :]
        if i == 0:  #diagonal of kernel matrix
            d[:, i : n+1] = kernel(x, x).T
        else:  #update the diagonal of the residual kernel matrix
            d[:, i : n+1] = kernel(x, x).squeeze() - np.sum(G[i : n+1, : i]**2, 1).T

        dtrace = np.sum(d[:, i:n+1])

        if dtrace <= 0:
            print("Warning: negative diagonal entry: ", diag)

        if dtrace <= precision:
            G = G[:,  :i] 
            subset = subset[ :i]
            break

        m2 = np.max(d[:,i : n+1])  #find the new best element
        j = np.argmax(d[:, i : n+1])
        #j = j + i - 1  #take into account the offset i
        j = j + i   #take into account the offset i
        m1 = sqrt(m2)
        subset[0, i] = j

        perm[ [i, j] ] = perm[ [j, i] ] #permute elements i and j
        #permute rows i and j
        G[[i, j], :i] = G[[j, i], :i]
        G[i, i] = m1  #new diagonal element

        #Calculate the i-th column. May 
        #introduce a slight numerical error compared to explicit calculation.

        G[i+1 : n +1, i] = (kernel(X[perm[i], :], X[perm[i+1:n+1], :]).T -
                            dot(G[i+1:n+1, :i], G[i, :i].T) )/ m1

    ind = argsort(perm)
    G = G[ind, :]
    return G

