Welcome to multiview-ndd's documentation!
=========================================

.. toctree::
   :maxdepth: 4

multiview.construct
===================

.. currentmodule:: multiview.construct

Random Gaussian Projection
--------------------------

.. autofunction:: random_gaussian_projection

Read more about sklearn's implementation `here <https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.GaussianRandomProjection.html>`_.

Random Subspace Method
----------------------

.. autofunction:: random_subspace_method

multiview.cotraining
====================

.. currentmodule:: multiview.cotraining

Cotraining Classifier
---------------------

.. autoclass:: CTClassifier
    :exclude-members: get_params, set_params

multiview.embed
===============

.. currentmodule:: multiview.embed

Generalized Canonical Correlation Analysis
------------------------------------------

.. autoclass:: GCCA
    :exclude-members: get_params, set_params

Omnibus Embedding
-----------------

.. autoclass:: Omnibus
    :exclude-members: transform, get_params, set_params

Partial Least Squares Regression
--------------------------------

.. autofunction:: partial_least_squares_embedding

Multiview Multidimensional Scaling
----------------------------------

.. autoclass:: MVMDS
    :exclude-members: transform, get_params, set_params

Split Autoencoder
-----------------

.. autoclass:: SplitAE
    :exclude-members: get_params, set_params


multiview.cluster
=================

.. currentmodule:: multiview.cluster

Multiview Spectral Clustering
-----------------------------

.. autoclass:: MultiviewSpectralClustering
    :exclude-members: get_params, set_params

Multiview K Means
-----------------

.. autoclass:: MultiviewKMeans
    :exclude-members: get_params, set_params







Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
