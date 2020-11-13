Changelog
=========

.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |MajorFeature| replace:: :raw-html:`<font color="green">[Major Feature]</font>`
.. |Feature| replace:: :raw-html:`<font color="green">[Feature]</font>`
.. |Efficiency| replace:: :raw-html:`<font color="blue">[Efficiency]</font>`
.. |Enhancement| replace:: :raw-html:`<font color="blue">[Enhancement]</font>`
.. |Fix| replace:: :raw-html:`<font color="red">[Fix]</font>`
.. |API| replace:: :raw-html:`<font color="DarkOrange">[API]</font>`

Change tags (adopted from `sklearn <https://scikit-learn.org/stable/whats_new/v0.23.html>`_):

- |MajorFeature| : something big that you couldn’t do before. 

- |Feature| : something that you couldn’t do before.

- |Efficiency| : an existing feature now may not require as much computation or memory.

- |Enhancement| : a miscellaneous minor improvement.

- |Fix| : something that previously didn’t work as documentated – or according to reasonable expectations – should now work.

- |API| : you will need to change your code to have the same effect in the future; or a feature will be removed in the future.

Version 0.4.0
-------------
**In development**.

Updates in this release:

`mvlearn.compose <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/compose>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |MajorFeature| Adds an ``mvlearn.compose`` module with Merger and Splitter objects to create single views from multiviews and vice versa: ``ConcatMerger``, ``AverageMerger``, and ``SimpleSplitter``. `#228 <https://github.com/mvlearn/mvlearn/pull/228>`_, `#234 <https://github.com/mvlearn/mvlearn/pull/234>`_ by `Pierre Ablin`_.
- |MajorFeature| Adds ``ViewTransformer`` to apply a single view transformer to each view separately. `#229 <https://github.com/mvlearn/mvlearn/pull/229>`_ by `Pierre Ablin`_, `#263 <https://github.com/mvlearn/mvlearn/pull/263>`_ by `Ronan Perry`_.
- |MajorFeature| Adds ``ViewClassifier`` to apply a single view classifier to each view separately. `#263 <https://github.com/mvlearn/mvlearn/pull/263>`_ by `Ronan Perry`_.
- |Feature| Switches ``random_subspace_method`` and ``random_gaussian_projection`` functions to sklearn-compliant estimators ``RandomSubspaceMethod`` and ``RandomGaussianProjection``. `#263 <https://github.com/mvlearn/mvlearn/pull/263>`_ by `Ronan Perry`_.
- |API| The ``mvlearn.construct`` module was merged into ``mvlearn.compose`` due to overlapping functionality. Any imports statements change accordingly. `#258 <https://github.com/mvlearn/mvlearn/pull/258>`_ by `Ronan Perry`_.

`mvlearn.construct <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/construct>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |API| The ``mvlearn.construct`` module was merged into ``mvlearn.compose`` due to overlapping functionality and no longer exists. Any imports statements change accordingly. `#258 <https://github.com/mvlearn/mvlearn/pull/258>`_ by `Ronan Perry`_.

`mvlearn.decomposition <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/decomposition>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |Feature| Adds ``GroupICA`` and ``GroupPCA``. `#225 <https://github.com/mvlearn/mvlearn/pull/225>`_ by `Pierre Ablin`_ and `Hugo Richard <https://github.com/hugorichard>`_.

`mvlearn.embed <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/embed>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |Feature| Adds Multi CCA (``MCCA``) and Kernel MCCA (``KMCCA``) for two or more views. `#249 <https://github.com/mvlearn/mvlearn/pull/249>`_ by `Ronan Perry`_ and `Iain Carmichael`_.

`mvlearn.model_selection <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/model_selection>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |MajorFeature| Adds an ``model_selection`` module with multiview cross validation. `#234 <https://github.com/mvlearn/mvlearn/pull/234>`_ by `Pierre Ablin`_.

- |Feature| Adds the function ``model_selection.train_test_split`` to wrap that of `sklearn <scikit-learn <https://scikit-learn.org/>`_ for multiview data or items. `#174 <https://github.com/mvlearn/mvlearn/pull/174>`_ by `Alexander Chang <https://github.com/achang63>`_ and `Gavin Mischler <https://gavinmischler.github.io/>`_.

`mvlearn.utils <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/utils>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |Enhancement| Adds a parameter to utils.check_Xs so that the function also returns the dimensions (n_views, n_samples, n_features) of the input dataset. `#235 <https://github.com/mvlearn/mvlearn/pull/235>`_ by `Pierre Ablin`_.

Version 0.3.0
-------------
Updates in this release:

- ``cotraining`` module changed to ``semi_supervised``.
- ``factorization`` module changed to ``decomposition``.
- A new class within the ``semi_supervised`` module, ``CTRegressor``, and regression tool for 2-view semi-supervised learning, following the cotraining framework.
- Three multiview ICA methods added: MultiviewICA, GroupICA, PermICA with ``python-picard`` dependency.
- Added parallelizability to GCCA using joblib and added ``partial_fit`` function to handle streaming or large data.
- Adds a function (get_stats()) to perform statistical tests within the ``embed.KCCA`` class so that canonical correlations and canonical variates can be robustly. assessed for significance. See the documentation in Reference for more details.
- Adds ability to select which views to return from the UCI multiple features dataset loader, ``datasets.UCI_multifeature``.
- API enhancements including base classes for each module and algorithm type, allowing for greater flexibility to extend ``mvlearn``.
- Internals of ``SplitAE`` changed to snake case to fit with the rest of the package.
- Fixes a bug which prevented the ``visualize.crossviews_plot`` from plotting when each view only has a single feature.
- Changes to the ``mvlearn.datasets.gaussian_mixture.GaussianMixture`` parameters to better mimic sklearn's datasets.
- Fixes a bug with printing error messages in a few classes.


Patch 0.2.1
-----------
Fixed missing ``__init__.py`` file in the ``ajive_utils`` submodule.

Version 0.2.0
-------------
Updates in this release:

- ``MVMDS`` can now also accept distance matrices as input, rather than only views of data with samples and features
- A new clustering algorithm, ``CoRegMultiviewSpectralClustering`` - co-regularized multi-view spectral clustering functionality
- Some attribute names slightly changed for more intuitive use in ``DCCA``, ``KCCA``, ``MVMDS``, ``CTClassifier``
- Option to use an Incomplete Cholesky Decomposition method for ``KCCA`` to reduce up computation times
- A new module, ``factorization``, containing the ``AJIVE`` algorithm - angle-based joint and individual variance explained
- Fixed issue where signal dimensions of noise were dependent in the GaussianMixtures class
- Added a dependecy to ``joblib`` to enable parallel clustering implementation
- Removed the requirements for ``torchvision`` and ``pillow``, since they are only used in tutorials


Version 0.1.0
-------------

We’re happy to announce the first major stable version of ``mvlearn``.
This version includes multiple new algorithms, more utility functions, as well as significant enhancements to the documentation. Here are some highlights of the big updates.

- Deep CCA, (``DCCA``) in the ``embed`` module
- Updated ``KCCA`` with multiple kernels
- Synthetic multi-view dataset generator class, ``GaussianMixture``, in the ``datasets`` module
- A new module, ``plotting``, which includes functions for visualizing multi-view data, such as ``crossviews_plot`` and ``quick_visualize``
- More detailed tutorial notebooks for all algorithms

Additionally, mvlearn now makes the ``torch`` and ``tqdm`` dependencies optional, so users who don’t need the DCCA or SplitAE functionality do not have to import such a large package. **Note** this is only the case for installing with pip. Installing from ``conda`` includes these dependencies automatically. To install the full version of mvlearn with ``torch`` and ``tqdm`` from pip, you must include the optional torch in brackets:

    .. code-block:: python

        pip3 install mvlearn[torch]

or

    .. code-block:: python

        pip3 install --upgrade mvlearn[torch]


To install **without** ``torch``, do:

    .. code-block:: python

        pip3 install mvlearn

or

    .. code-block:: python

        pip3 install --upgrade mvlearn



.. _Pierre Ablin: https://pierreablin.com/
.. _Ronan Perry: http://rflperry.github.io/
.. _Iain Carmichael: https://idc9.github.io/