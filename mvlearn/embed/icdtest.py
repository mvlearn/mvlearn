import pandas as pd
from numpy import *
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
#import sys
#np.set_printoptions(threshold=sys.maxsize)

X = pd.read_csv("x1.csv", header=None).values
X2 = pd.read_csv("x2.csv", header=None).values

def _make_kernel(X, Y, ktype, constant=0.1, degree=2.0, sigma=1.0):
    Nl = len(X)
    Nr = len(Y)
    N0l = np.eye(Nl) - 1 / Nl * np.ones(Nl)
    N0r = np.eye(Nr) - 1 / Nr * np.ones(Nr)

    # Linear kernel
    if ktype == "linear":
        return N0l @ (X @ Y.T) @ N0r

    # Polynomial kernel
    elif ktype == "poly":
        return N0l @ (X @ Y.T + constant) ** degree @ N0r

    # Gaussian kernel
    elif ktype == "gaussian":
        distmat = euclidean_distances(X, Y, squared=True)

        return np.exp(-distmat / (2 * sigma ** 2))
# =============================================================================
#         j1 = np.ones((X.shape[0], 1))
#         j2 = np.ones((Y.shape[0], 1))
# 
#         diagK1 = np.sum(X**2, 1)
#         diagK2 = np.sum(Y**2, 1)
# 
#         X1X2 = np.dot(X, Y.T)
# 
#         Q = (2*X1X2 - np.outer(diagK1, j2) - np.outer(j1, diagK2) )/ (2*sigma**2)
# 
#         return np.exp(Q)
# =============================================================================
    
    # Gaussian diagonal kernel
    elif ktype == "gaussian-diag":
        # N0l@@N0r
        return np.exp(-np.sum(np.power((X-Y), 2),axis=1)/(2*sigma**2))

N = len(X)
precision = 0.000001
mrank = 50
sigma = 1.0
 
perm = np.arange(N)
d = np.zeros(N)
G = np.zeros((N, mrank))
subset = np.zeros(mrank)

for i in range(mrank):
    x_new = X[perm[i:N+1], :]
    if i == 0:
        d[i:N+1] = _make_kernel(x_new, x_new, "gaussian-diag").T
    else:
        fk2 = _make_kernel(x_new, x_new, "gaussian-diag").T
        fk = np.sum(np.power(G[i:N+1, :i], 2), axis=1).T
        d[i:N+1] = (fk2 - fk)
    
    #dtrace stuff
    
    j = np.argmax(d[i:N+1])
    m2 = np.max(d[i:N+1])
    j = j+i
    m1 = np.sqrt(m2)
    subset[i] = j
    
    perm[ [i, j] ] = perm[ [j, i] ]
    G[[i, j], :i] = G[[j, i], :i]
    G[i, i] = m1
    
    z1 = _make_kernel([X[perm[i], :]], X[perm[i+1:N+1], :], "gaussian", sigma)
    z2 =(G[i+1:N+1, :i]@(G[i, :i].T))
    
    G[i+1:N+1, i] = (z1 - z2)/m1

ind = np.argsort(perm)
G = G[ind, :]

# =============================================================================
# n = X.shape[0]
# m = 50
# perm = np.arange(n)  #permutation vector
# d = np.zeros((1, n))  #diagonal of the residual kernel matrix
# G = np.zeros((n, m))
# subset = np.zeros((1, m))
# precision =0.0000001
# 
# for i in range(1):
#     x = X[perm[i : n+1], :]
#     if i == 0:  #diagonal of kernel matrix
#         d[:, i : n+1] = _make_kernel(x, x, "gaussian-diag").T
#     else:  #update the diagonal of the residual kernel matrix
#         d[:, i : n+1] = _make_kernel(x, x, "gaussian-diag").squeeze() - np.sum(G[i : n+1, : i]**2, 1).T
# 
#     dtrace = np.sum(d[:, i:n+1])
# 
# # =============================================================================
# #     if dtrace <= 0:
# #         print("Warning: negative diagonal entry: ", diag)
# # =============================================================================
#     if dtrace <= precision:
#         G = G[:,  :i] 
#         subset = subset[ :i]
#         break
# 
#     m2 = np.max(d[:,i : n+1])  #find the new best element
#     j = np.argmax(d[:, i : n+1])
#     #j = j + i - 1  #take into account the offset i
#     j = j + i   #take into account the offset i
#     m1 = np.sqrt(m2)
#     subset[0, i] = j
# 
#     perm[ [i, j] ] = perm[ [j, i] ] #permute elements i and j
#     #permute rows i and j
#     G[[i, j], :i] = G[[j, i], :i]
#     G[i, i] = m1  #new diagonal element
# 
#     #Calculate the i-th column. May 
#     #introduce a slight numerical error compared to explicit calculation.
#     if i == 0:
#         print(G[i+1:n+1, :i].shape)
#         print(G[i, :i+1].T.shape)
#         G[i+1 : n +1, i] = (_make_kernel(X[perm[i], :].reshape(1, -1), X[perm[i+1:n+1], :], "gaussian").T -
#                             G[i+1:n+1, :i]@G[i, :i+1].T)/ m1
#     else:
#         G[i+1 : n +1, i] = (_make_kernel(X[perm[i], :], X[perm[i+1:n+1], :], "gaussian").T -
#                             G[i+1:n+1, :i]@G[i, :i].T)/ m1
# 
# ind = np.argsort(perm)
# G = G[ind, :]
# =============================================================================

# =============================================================================
# N = len(X)
# precision = 0.000001
# mrank = 50
# sigma = 1.0
#  
# perm = np.arange(N)
# d = np.zeros(N)
# G = np.zeros((N, mrank))
# subset = np.zeros(mrank)
# 
# for i in range(0):
#     x_new = X[perm[i:N], :]
#     if i == 0:
#         d[i:N] = _make_kernel(x_new, x_new, "gaussian-diag").T
#     else:
#         fk2 = _make_kernel(x_new, x_new, "gaussian-diag").T
#         fk = np.sum(np.power(G[i:N, :i], 2), axis=1).T
#         d[i:N] = (fk2 - fk)
# 
#     #dtrace stuff
# 
#     j = np.argmax(d[i:N])
#     m2 = np.max(d[i:N])
#     j = j+i
#     m1 = np.sqrt(m2)
#     subset[i] = j
#     
#     perm[i], perm[j] = perm[j], perm[i]
#     G[i, :i], G[j, :i] = G[j, :i], G[i, :i]
#     G[i, i] = m1
# 
#     z1 = _make_kernel([X[perm[i], :]], X[perm[i+1:N], :], "gaussian", sigma)
#     z2 =(G[i+1:N, :i]@(G[i, :i].T))
#     
#     G[i+1:N, i] = (z1 - z2)/m1
# ind = np.argsort(perm)
# G = G[ind, :]
# =============================================================================
    
# =============================================================================
# for i in range(50):
#     x_new = x[perm[i:N+1], :]
#     if i == 0:
#         d[i:N+1] = _make_kernel(x_new, x_new, "gaussian-diag").T
#     else:
#         fk2 = _make_kernel(x_new, x_new, "gaussian-diag").T
#         fk = np.sum(np.power(G[i:N+1, :i], 2), axis=1).T
#         d[i:N+1] = (fk2 - fk)
# 
# # =============================================================================
# #     dtrace = np.sum(d[:, i:N])
# #  
# #     if dtrace <= 0:
# #         print("Warning: negative diagonal entry: ")
# #  
# #     if dtrace <= precision:
# #         G = G[:,  :i] 
# #         subset = subset[ :i]
# #         break
# # =============================================================================
# 
#     j = np.argmax(d[i:N+1])
#     m2 = np.max(d[i:N+1])
#     j = j+i
#     m1 = np.sqrt(m2)
#     subset[i] = j
#     
#     perm[i], perm[j] = perm[j], perm[i]
#     G[i, :i], G[j, :i] = G[j, :i], G[i, :i]
#     G[i, i] = m1
# 
#     z1 = _make_kernel([x[perm[i], :]], x[perm[i+1:N+1], :], "gaussian", sigma)
#     z2 =(G[i+1:N+1, :i]@(G[i, :i].T))
#     
#     G[i+1:N+1, i] = (z1 - z2)/m1
# 
# ind = np.argsort(perm)
# G = G[ind, :]
# 
# =============================================================================

# =============================================================================
# n = X.shape[0]
# m = 50
# precision = 0.000001
# 
# perm = np.arange(n)  #permutation vector
# d = np.zeros((1, n))  #diagonal of the residual kernel matrix
# G = np.zeros((n, m))
# subset = np.zeros((1, m))
# 
# for i in range(m):
#     x = X[perm[i : n+1], :]
#     if i == 0:  #diagonal of kernel matrix
#         d[:, i : n+1] = _make_kernel(x, x, 'guassian-diag', 1)
#     else:  #update the diagonal of the residual kernel matrix
#         d[:, i : n+1] = _make_kernel(x, x, 'guassian-diag', 1).squeeze() - np.sum(G[i : n+1, : i]**2, 1)
# 
#     dtrace = np.sum(d[:, i:n+1])
# 
#     if dtrace <= 0:
#         print("Warning: negative diagonal entry: ")
# 
#     if dtrace <= precision:
#         G = G[:,  :i] 
#         subset = subset[ :i]
#         break
# 
#     m2 = np.max(d[:,i : n+1])  #find the new best element
#     j = np.argmax(d[:, i : n+1])
#     #j = j + i - 1  #take into account the offset i
#     j = j + i   #take into account the offset i
#     m1 = np.sqrt(m2)
#     subset[0, i] = j
# 
#     perm[ [i, j] ] = perm[ [j, i] ] #permute elements i and j
#     #permute rows i and j
#     G[[i, j], :i] = G[[j, i], :i]
#     G[i, i] = m1  #new diagonal element
# 
#     #Calculate the i-th column. May 
#     #introduce a slight numerical error compared to explicit calculation.
# 
#     G[i+1 : n +1, i] = (_make_kernel(X[perm[i], :], X[perm[i+1:n+1], :], 'guassian', 1) -
#                         np.dot(G[i+1:n+1, :i], G[i, :i].T) )/ m1
# 
# ind = np.argsort(perm)
# G = G[ind, :]
# =============================================================================


# =============================================================================
# for i in range(4):
#     x_new = X[perm[i:N], :]
#     if i == 0:
#         d[i:N] = _make_kernel(x_new, x_new, "gaussian-diag").T
#     else:
#         fk2 = _make_kernel(x_new, x_new, "gaussian-diag").T
#         fk = np.sum(np.power(G[i:N, :i], 2), axis=1).T
#         d[i:N] = (fk2 - fk)
#     
#     #dtrace stuff
#     
#     j = np.argmax(d[i:N])
#     m2 = np.max(d[i:N])
#     j = j+i
#     m1 = np.sqrt(m2)
#     subset[i] = j
#     
#     perm[i], perm[j] = perm[j], perm[i]
#     G[i, :i], G[j, :i] = G[j, :i], G[i, :i]
#     G[i, i] = m1
#     
#     z1 = _make_kernel([X[perm[i], :]], X[perm[i+1:N], :], "gaussian", sigma)
#     z2 =(G[i+1:N, :i]@(G[i, :i].T))
#     
#     G[i+1:N, i] = (z1 - z2)/m1
# =============================================================================
# =============================================================================
# i = 0
# x_new = X[perm[i:N], :]
# if i == 0:
#     d[i:N] = _make_kernel(x_new, x_new, "gaussian-diag").T
# else:
#     fk2 = _make_kernel(x_new, x_new, "gaussian-diag").T
#     fk = np.sum(np.power(G[i:N, :i], 2), axis=1).T
#     d[i:N] = (fk2 - fk)
# 
# #dtrace stuff
# 
# j = np.argmax(d[i:N+1]) 
# m2 = np.max(d[i:N])
# j = j+i
# m1 = np.sqrt(m2)
# subset[i] = j
# 
# perm[ [i, j] ] = perm[ [j, i] ]
# G[[i, j], :i] = G[[j, i], :i]
# G[i, i] = m1
# 
# z1 = _make_kernel([X[perm[i], :]], X[perm[i+1:N], :], "gaussian", sigma)
# z2 =(G[i+1:N, :i]@(G[i, :i].T))
# 
# G[i+1:N, i] = (z1 - z2)/m1
# 
# ######
# i = 1
# x_new = X[perm[i:N], :]
# if i == 0:
#     d[i:N] = _make_kernel(x_new, x_new, "gaussian-diag").T
# else:
#     fk2 = _make_kernel(x_new, x_new, "gaussian-diag").T
#     fk = np.sum(np.power(G[i:N, :i], 2), axis=1).T
#     d[i:N] = (fk2 - fk)
# 
# #dtrace stuff
# 
# j = np.argmax(d[i:N])
# m2 = np.max(d[i:N])
# j = j+i
# m1 = np.sqrt(m2)
# subset[i] = j
# 
# perm[i], perm[j] = perm[j], perm[i]
# G[i, :i], G[j, :i] = G[j, :i], G[i, :i]
# G[i, i] = m1
# 
# z1 = _make_kernel([X[perm[i], :]], X[perm[i+1:N], :], "gaussian", sigma)
# z2 =(G[i+1:N, :i]@(G[i, :i].T))
# 
# G[i+1:N, i] = (z1 - z2)/m1
# 
# ######
# i = 2
# x_new = X[perm[i:N], :]
# if i == 0:
#     d[i:N] = _make_kernel(x_new, x_new, "gaussian-diag").T
# else:
#     fk2 = _make_kernel(x_new, x_new, "gaussian-diag").T
#     fk = np.sum(np.power(G[i:N, :i], 2), axis=1).T
#     d[i:N] = (fk2 - fk)
# 
# #dtrace stuff
# 
# j = np.argmax(d[i:N])
# m2 = np.max(d[i:N])
# j = j+i
# m1 = np.sqrt(m2)
# subset[i] = j
# 
# perm[i], perm[j] = perm[j], perm[i]
# G[i, :i], G[j, :i] = G[j, :i], G[i, :i]
# G[i, i] = m1
# 
# z1 = _make_kernel([X[perm[i], :]], X[perm[i+1:N], :], "gaussian", sigma)
# z2 =(G[i+1:N, :i]@(G[i, :i].T))
# 
# G[i+1:N, i] = (z1 - z2)/m1
# 
# #######
# i = 3
# x_new = X[perm[i:N], :]
# if i == 0:
#     d[i:N] = _make_kernel(x_new, x_new, "gaussian-diag").T
# else:
#     fk2 = _make_kernel(x_new, x_new, "gaussian-diag").T
#     fk = np.sum(np.power(G[i:N, :i], 2), axis=1).T
#     d[i:N] = (fk2 - fk)
# 
# #dtrace stuff
# 
# j = np.argmax(d[i:N]) #here is the divergence!
# m2 = np.max(d[i:N])
# j = j+i
# m1 = np.sqrt(m2)
# subset[i] = j
# 
# perm[i], perm[j] = perm[j], perm[i]
# G[i, :i], G[j, :i] = G[j, :i], G[i, :i]
# G[i, i] = m1
# 
# z1 = _make_kernel([X[perm[i], :]], X[perm[i+1:N], :], "gaussian", sigma)
# z2 =(G[i+1:N, :i]@(G[i, :i].T))
# 
# G[i+1:N, i] = (z1 - z2)/m1
# =============================================================================

#ind = np.argsort(perm)
#G[ind, :]


