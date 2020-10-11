"""
===============
Deep CCA (DCCA)
===============
"""

from mvlearn.embed import DCCA
from mvlearn.datasets import GaussianMixture
from mvlearn.plotting import crossviews_plot
from mvlearn.model_selection import train_test_split
import numpy as np

###############################################################################
# Polynomial-Transformed Latent Correlation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Latent variables are sampled from two multivariate Gaussians with equal
# prior probability. Then a polynomial transformation is applied and noise is
# added independently to both the transformed and untransformed latents.


n_samples = 2000
means = [[0, 1], [0, -1]]
covariances = [np.eye(2), np.eye(2)]
gm = GaussianMixture(n_samples, means, covariances, random_state=42,
                     shuffle=True, shuffle_random_state=42)
latent, y = gm.get_Xy(latents=True)

# The latent data is plotted against itself to reveal the underlying
# distribtution.

crossviews_plot([latent, latent], labels=y,
                title='Latent Variable', equal_axes=True)

# The noisy latent variable (view 1) is plotted against the transformed latent
# variable (view 2), an example of a dataset with two views.


# Split data into train and test sets
Xs, y = gm.sample_views(transform='poly', n_noise=2).get_Xy()
Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3,
                                                      random_state=42)

crossviews_plot(Xs_test, labels=y_test,
                title='Testing Data View 1 vs. View 2 '
                      '(Polynomial Transform + noise)',
                equal_axes=True)

###############################################################################
# Fit DCCA model to uncover latent distribution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The output dimensionality is still 4.


# Define parameters and layers for deep model
features1 = Xs_train[0].shape[1]  # Feature sizes
features2 = Xs_train[1].shape[1]
layers1 = [256, 256, 4]  # nodes in each hidden layer and the output size
layers2 = [256, 256, 4]

dcca = DCCA(input_size1=features1, input_size2=features2, n_components=4,
            layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=500)
dcca.fit(Xs_train)
Xs_transformed = dcca.transform(Xs_test)

###############################################################################
# Visualize the transformed data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can see that it has uncovered the latent correlation between views.


crossviews_plot(Xs_transformed, labels=y_test,
                title='Transformed Testing Data View 1 vs. View 2 '
                      '(Polynomial Transform + noise)',
                equal_axes=True)

###############################################################################
# Sinusoidal-Transformed Latent Correlation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Following the same procedure as above, latent variables are sampled from two
# multivariate Gaussians with equal prior probability. This time, a sinusoidal
# transformation is applied and noise is added independently to both the
# transformed and untransformed latents.


n_samples = 2000
means = [[0, 1], [0, -1]]
covariances = [np.eye(2), np.eye(2)]
gm = GaussianMixture(n_samples, means, covariances, random_state=42,
                     shuffle=True, shuffle_random_state=42)

# Split data into train and test segments
Xs, y = gm.sample_views(transform='sin', n_noise=2).get_Xy()
Xs_train, Xs_test, y_train, y_test = train_test_split(Xs, y, test_size=0.3,
                                                      random_state=42)

crossviews_plot(Xs_test, labels=y_test,
                title='Testing Data View 1 vs. View 2 '
                      '(Polynomial Transform + noise)',
                equal_axes=True)

###############################################################################
# Fit DCCA model to uncover latent distribution
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# The output dimensionality is still 4.


# Define parameters and layers for deep model
features1 = Xs_train[0].shape[1]  # Feature sizes
features2 = Xs_train[1].shape[1]
layers1 = [256, 256, 4]  # nodes in each hidden layer and the output size
layers2 = [256, 256, 4]

dcca = DCCA(input_size1=features1, input_size2=features2, n_components=4,
            layer_sizes1=layers1, layer_sizes2=layers2, epoch_num=500)
dcca.fit(Xs_train)
Xs_transformed = dcca.transform(Xs_test)

###############################################################################
# Visualize the transformed data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We can see that it has uncovered the latent correlation between views.

crossviews_plot(Xs_transformed, labels=y_test,
                title='Transformed Testing Data View 1 vs. View 2 '
                      '(Sinusoidal Transform + noise)',
                equal_axes=True)
