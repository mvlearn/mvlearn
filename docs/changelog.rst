Changelog
=========

Change tags (adopted from `sklearn <https://scikit-learn.org/stable/whats_new/v0.23.html>`_):

- |Major Feature| : something big that you couldn’t do before.

- |Feature| : something that you couldn’t do before.

- |Efficiency| : an existing feature now may not require as much computation or memory.

- |Enhancement| : a miscellaneous minor improvement.

- |Fix| : something that previously didn’t work as documentated – or according to reasonable expectations – should now work.

- |API Change| : you will need to change your code to have the same effect in the future; or a feature will be removed in the future.

Version 0.4.0
-------------
**In development**. 

Updates in this release:

`mvlearn.decomposition <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/decomposition>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |Feature| Adds GroupICA and GroupPCA. `#225 <https://github.com/mvlearn/mvlearn/pull/225>`_ by `Pierre Ablin <https://github.com/pierreablin>`_ and `Hugo Richard <https://github.com/hugorichard>`_.

`mvlearn.merge <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/merge>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |Major Feature| Introduces Merger objects to create single views from multiviews: ConcatMerger and AverageMerger. `#228 <https://github.com/mvlearn/mvlearn/pull/228>`_ by `Pierre Ablin <https://github.com/pierreablin>`_.

`mvlearn.preprocessing <https://github.com/mvlearn/mvlearn/tree/master/mvlearn/preprocessing>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- |Major Feature| Adds a preprocessing module with ViewTransformer to apply a single view function to each view separately. `#229 <https://github.com/mvlearn/mvlearn/pull/229>`_ by `Pierre Ablin <https://github.com/pierreablin>`_.

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
