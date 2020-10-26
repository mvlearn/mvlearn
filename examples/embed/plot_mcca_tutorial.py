"""
======================
Multiview CCA Tutorial
======================

This tutorial demonstrates the use of multiview CCA, which allows for greater
than two views. As is demonstrated, it allows for both the addition of
regularization as well as the use of a kernel to compute the distance
(Gram) matrices.

"""

# Authors: Iain Carmichael, Ronan Perry

from mvlearn.datasets import sample_joint_factor_model
from mvlearn.embed import MCCA, KMCCA
from mvlearn.plotting import crossviews_plot


n_views = 3
n_samples = 1000
n_features = [10, 20, 30]
joint_rank = 3

# sample data from a joint factor model
# m, noise_std control the difficulty of the problem
Xs, U_true, Ws_true = sample_joint_factor_model(
    n_views=n_views, n_samples=n_samples, n_features=n_features,
    joint_rank=joint_rank, m=5, noise_std=1, random_state=23,
    return_decomp=True)

###############################################################################
# Multiview CCA (MCCA)
# ^^^^^^^^^^^^^^^^^^^^

# the default is no regularization meaning this is SUMCORR-AVGVAR MCCA
mcca = MCCA(n_components=joint_rank)

# the fit-transform method outputs the scores for each view
mcca_scores = mcca.fit_transform(Xs)
crossviews_plot(mcca_scores[[0, 1]],
                title='MCCA embedding dimensions of the first two views',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})

# MCCA with regularization
#
# We can add regularization with the `regs` argument
# to handle high-dimensional data.


# regularization value of .1 for each view
mcca = MCCA(n_components=joint_rank, regs=.5).fit(Xs)

# the fit-transform method outputs the scores for each view
mcca_scores = mcca.fit_transform(Xs)
crossviews_plot(mcca_scores[[0, 1]],
                title='MCCA embedding with regularization',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})


###############################################################################
# Informative MCCA: PCA then MCCA
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can also handle high-dimensional data with i-MCCA. We first compute a
# low rank PCA for each view, then run MCCA on the reduced data.


# i-MCCA where we first extract the first 5 PCs from each data view
mcca = MCCA(n_components=joint_rank, signal_ranks=[5, 5, 5]).fit(Xs)
mcca_scores = mcca.transform(Xs)
crossviews_plot(mcca_scores[[0, 1]],
                title='PCA-MCCA embeddings',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})


###############################################################################
# Kernel MCCA
# ^^^^^^^^^^^
#
# We can compute kernel MCCA with the KMCCA() object.


# fit kernel MCCA with a linear kernel
kmcca = KMCCA(n_components=joint_rank, kernel='linear').fit(Xs)
kmcca_scores = kmcca.transform(Xs)
crossviews_plot(kmcca_scores[[0, 1]],
                title='KMCCA embeddings',
                equal_axes=True,
                scatter_kwargs={'alpha': 0.4, 's': 2.0})
