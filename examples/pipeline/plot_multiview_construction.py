"""
=======================================================
Constructing multiple views to classify singleview data
=======================================================

As demonstrated in "Asymmetric bagging and random subspace for support vector
machines-based relevance feedback in image retrieval" (Dacheng 2006), in high
dimensional data it can be useful to subsample the features and construct
multiple classifiers on each subsample whose individual predictions are
combined using majority vote. This is akin to bagging but concerns the
features rather than samples and is how random forests are ensembled
from individual decision trees. Here, we apply Linear Discriminant Analysis
(LDA) to a high dimensional image classification problem and demonstrate
how subsampling features can help when the sample size is relatively low.

A variety of possible subsample dimensions are considered, and for each the
number of classifiers (views) is chosen such that their product is equal to
the number of features in the singleview data.

Two subsampling methods are applied. The random subspace method simply selects
a random subset of the features. The random Gaussian projection method creates
new features by sampling random multivariate Gaussian vectors used to project
the original features. The latter method can potentially help in nonlinear
settings where combinations of features better capture informative relations.
"""

# Author: Ronan Perry
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.datasets import fetch_olivetti_faces
from mvlearn.compose import RandomSubspaceMethod, RandomGaussianProjection, \
    ViewClassifier


# Load the singleview Olivevetti faces dataset from sklearn
X, y = fetch_olivetti_faces(return_X_y=True)

rsm_scores = []
rgp_scores = []
clf_scores = []

# The data has 4096 features. The following subspace dimensions are used
dims = [16, 64, 256, 1024]

skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
skf.get_n_splits(X, y)

for i, (test_index, train_index) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    rsm_scores.append([])
    rgp_scores.append([])
    for dim in dims:
        n_views = int(X.shape[1] / dim)

        rsm_clf = make_pipeline(
            StandardScaler(),
            RandomSubspaceMethod(n_views=n_views, subspace_dim=dim),
            ViewClassifier(LinearDiscriminantAnalysis())
        )
        rsm_scores[i].append(
            rsm_clf.fit(X_train, y_train).score(X_test, y_test))

        rgp_clf = make_pipeline(
            StandardScaler(),
            RandomGaussianProjection(n_views=n_views, n_components=dim),
            ViewClassifier(LinearDiscriminantAnalysis())
        )
        rgp_scores[i].append(
            rgp_clf.fit(X_train, y_train).score(X_test, y_test))

    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    clf_scores.append(clf.fit(X_train, y_train).score(X_test, y_test))

fig, ax = plt.subplots()
ax.axhline(
    np.mean(clf_scores), ls='--', c='grey', label='LDA singleview score')
ax.axvline(
    X_train.shape[0], ls=':', c='grey', label='Number of training samples')
ax.errorbar(
    dims, np.mean(rsm_scores, axis=0),
    yerr=np.std(rsm_scores, axis=0), label='LDA o Random Subspace')
ax.errorbar(
    dims, np.mean(rgp_scores, axis=0),
    yerr=np.std(rgp_scores, axis=0), label='LDA o Random Gaussian Projection')
ax.set_xlabel('Number of subsampled dimensions')
ax.set_ylabel('Score')
plt.title('Classification accuracy using constructed multiview data')
plt.legend()
plt.show()
