"""
vs PCA
======

"""

from mvlearn.embed import GCCA
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from scipy.sparse.linalg import svds



def get_train_test(n=100, mu=0, var=1, var2=1, nviews=3,m=1000):
    # Creates train and test data with a
    # - shared signal feature ~ N(mu, var1)
    # - an independent noise feature ~ N(mu, var2)
    # - independent noise feautures ~ N(0, 1)
    np.random.seed(0)
   
    X_TRAIN = np.random.normal(mu,var,(n,1))
    X_TEST = np.random.normal(mu,var,(n,1))

    Xs_train = []
    Xs_test = []
    for i in range(nviews):
        X_train = np.hstack((np.random.normal(0,1,(n,i)),
                             X_TRAIN,
                             np.random.normal(0,1,(n,m-2-i)),
                             np.random.normal(0,var2,(n,1))
                            ))
        X_test = np.hstack((np.random.normal(0,1,(n,i)),
                            X_TEST,
                            np.random.normal(0,1,(n,m-2-i)),
                            np.random.normal(0,var2,(n,1))
                           ))
       
        Xs_train.append(X_train)
        Xs_test.append(X_test)
   
    return(Xs_train,Xs_test)

###############################################################################
# Positive Test
# -------------
###############################################################################
# Setting:
# ^^^^^^^^
# 1 high variance shared signal feature, 1 high variance noise feature


nviews = 3
Xs_train, Xs_test = get_train_test(var=10,var2=10,nviews=nviews,m=1000)



gcca = GCCA(n_components=2)
gcca.fit(Xs_train)
Xs_hat = gcca.transform(Xs_test)

###############################################################################
# Results:
# ^^^^^^^^^
# - GCCA results show high correlation on testing data


np.corrcoef(np.array(Xs_hat)[:,:,0])



Xs_hat = []
for i in range(len(Xs_train)):
    _,_,vt = svds(Xs_train[i],k=1)
    Xs_hat.append(Xs_test[i] @ vt.T)

# - PCA selects shared dimension but also high noise dimension and so weaker
# correlation on testing data


np.corrcoef(np.array(Xs_hat)[:,:,0])

###############################################################################
# Negative Test
# -------------
###############################################################################
# Setting:
# ^^^^^^^^
# 1 low variance shared feature


nviews = 3
Xs_train, Xs_test = get_train_test(var=1,var2=1,nviews=nviews,m=1000)



gcca = GCCA(n_components = 2)
gcca.fit(Xs_train)
Xs_hat = gcca.transform(Xs_test)

###############################################################################
# Results:
# ^^^^^^^^^
# - GCCA fails to select shared feature and so shows low correlation on
# testing data


np.corrcoef(np.array(Xs_hat)[:,:,0])



Xs_hat = []
for i in range(len(Xs_train)):
    _,_,vt = svds(Xs_train[i],k=1)
    Xs_hat.append(Xs_test[i] @ vt.T)

# - PCA fails to select shared feature and shows low correlation on testing
# data


np.corrcoef(np.array(Xs_hat)[:,:,0])

