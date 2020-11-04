"""
=========================================
Cross-view information retrieval with CCA
=========================================

MCCA and KMCCA are used to learn informative relationships
between 2 views of digits in the UCI multi feature dataset: 76 Fourier
coefficients of the digit shapes and 2x3 window pixel averages.
The Fourier view of a held out sample is transformed and matched
to the pixel averages view from the fitted data for which it has the
largest correlation in the transformed space.
"""

# Author: Ronan Perry
# License: MIT

from mvlearn.embed import MCCA, KMCCA
from mvlearn.datasets import load_UCImultifeature
import matplotlib.pyplot as plt
import numpy as np


###############################################################################
# Load and fit the UCI data
# -------------------------
#
# The CCAs learn transformations of the Fourier view and Pixel view that
# maximizes the cross-view correlation between samples. Here, we have a Fourier
# sample X1 with unknown pixels. We transform X1 based on the fitted model
# and find the closest transformed X2 among fitted pixels that in order
# to retrieve a guess as to the unknown pixels.

Xs, _ = load_UCImultifeature(views=[3, 0], shuffle=True, random_state=0)
Xs_test = [X[-1].reshape(1, -1) for X in Xs]
Xs_train = [X[:-1] for X in Xs]

# Fit and match MCCA
mcca = MCCA(n_components=5, regs=0.01).fit(Xs_train)

train_scores = mcca.transform_view(Xs_train[0], 0)
test_score = mcca.transform_view(Xs_test[1], 1)

match_index_mcca = np.argmax([np.corrcoef(test_score, score)[0, 1]
                              for score in train_scores])

# Fit and match KMCCA
kmcca = KMCCA(n_components=5, kernel='rbf', regs=0.01).fit(Xs_train)
train_scores = kmcca.transform_view(Xs_train[0], 0)
test_score = kmcca.transform_view(Xs_test[1], 1)

match_index_kmcca = np.argmax([np.corrcoef(test_score, score)[0, 1]
                               for score in train_scores])

###############################################################################
# Show selected match results
# ---------------------------
#
# As we can see, from all the digits 0-9, both MCCA and KMCCA retrieve a digit
# 3 image, which is the digit image of the true but missing view of the
# test sample. This is entirely unsupervised.

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 3))
ax1.imshow(Xs_test[0].reshape(16, 15))
ax1.set_title('True, unknown match')

ax2.imshow(Xs_train[0][match_index_mcca].reshape(16, 15))
ax2.set_title('MCCA selected match')

ax3.imshow(Xs_train[0][match_index_kmcca].reshape(16, 15))
ax3.set_title('KMCCA selected match')
plt.axis('off')
plt.show()
