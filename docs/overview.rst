Overview of mvlearn_
====================

.. _mvlearn: https://github.com/mvlearn/mvlearn

Motivation
----------

mvlearn aims to serve as a community-driven open-source software package that offers reference implementations for algorithms
and methods related to multiview learning, machine learning in settings where there are multiple incommensurate views or feature
sets for each sample. It brings together the most widely-used tools in this setting with a standardized scikit-learn like API,
well tested code and high-quality documentation. Doing so we aim to facilitate application, extension, comparison of methods, and
offer a foundation for research into new multiview algorithms. We welcome new contributors and the addition of methods with proven
efficacy and current use.

Background
----------

Multiview data, in which each sample is represented by multiple views of distinct features, are often seen in real-world data,
and related methods have grown in popularity. A view is defined as a partition of the complete set of feature variables
[#1xu_2013]_. Depending on the domain, these views may arise naturally from unique sources, or they may correspond to
subsets of the same underlying feature space. For example, a doctor may have an MRI scan, a CT scan, and the answers to a clinical
questionnaire for a diseased patient. However, classical methods for inference and analysis are often poorly suited to account for
multiple views of the same sample, since they cannot properly account for complementing views that hold differing statistical
properties [#2zhao_2017]_. To deal with this, many multiview learning methods have been developed to take advantage of multiple
data views and produce better results in various tasks [#3sun_2013]_ [#4hardoon_2004]_ [#5chao_2017]_ [#6yang_2014]_.

Examples
--------

Brief examples
^^^^^^^^^^^^^^

-  Import mvlearn

    .. code:: python

        import mvlearn


- Decompose two views using multiview PCA to capture joint information

    .. code:: python

        from mvlearn.decomposition import GroupPCA
        # X1 and X2 are data matrices, each with n samples
        Xs = [X1, X2] # multiview data
        Xs_components = GroupPCA().fit_transform(Xs)

- Cluster two views using multiview KMeans to find shared labels

    .. code:: python

        from mvlearn.cluster import MultiviewKMeans
        # X1 and X2 are data matrices, each with n samples
        Xs = [X1, X2] # multiview data
        labels = MultiviewKMeans().fit_predict(Xs)


Highlighted full examples
^^^^^^^^^^^^^^^^^^^^^^^^^

- `Nutrimouse dataset case study <auto_examples/datasets/plot_nutrimouse.html>`_:
    A collection of multiview learning methods across modules provide insights to a 2-view genomics dataset.

- `Multiview vs singleview clustering on the UCI multiview digits <auto_examples/cluster/plot_mv_vs_singleview_spectral.html>`_:
    Multiview clustering strongly outperforms single view clustering on a multiview dataset of handwritten digits.

- `A comparison of CCA algorithms <auto_examples/embed/plot_cca_comparison.html>`_:
    Canonical correlation analysis (CCA) variants find linearly correlated projections of each view. Linear and nonlinear
    variants are compared in various simulated settings.

Python
------

Python is a powerful programming language that allows concise expressions of network
algorithms.  Python has a vibrant and growing ecosystem of packages that
mvlearn uses to provide more features such as numerical linear algebra. In order to make the most out of mvlearn you will want to know how
to write basic programs in Python.  Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_.

Currently, mvlearn is supported for Python 3.6, 3.7, and 3.8.

Free software
-------------

mvlearn is free software; you can redistribute it and/or modify it under the
terms of the :doc:`MIT License </license>`.  We welcome contributions.
Join us on `GitHub <https://github.com/mvlearn/mvlearn>`_.

History
-------

mvlearn was developed during the end of 2019 by Richard Guo, Ronan Perry, Gavin Mischler, Theo Lee, Alexander Chang, Arman Koul, and Cameron Franz, a team out of the Johns Hopkins University NeuroData group.

Citing `mvlearn`
----------------

If you find the package useful for your research, please cite our `JMLR Paper <https://www.jmlr.org/papers/volume22/20-1370/20-1370.pdf>`_.

Perry, Ronan, et al. "mvlearn: Multiview Machine Learning in Python." Journal of Machine Learning Research 22.109 (2021): 1-7.

BibTeX entry:

.. code:: tex

    @article{perry2021mvlearn,
      title={mvlearn: Multiview Machine Learning in Python},
      author={Perry, Ronan and Mischler, Gavin and Guo, Richard and Lee, Theodore and Chang, Alexander and Koul, Arman and Franz, Cameron and Richard, Hugo and Carmichael, Iain and Ablin, Pierre and Gramfort, Alexandre and Vogelstein, Joshua T.},
      journal={Journal of Machine Learning Research},
      volume={22},
      number={109},
      pages={1--7},
      year={2021}
    }

References
----------

.. [#1xu_2013] Chang Xu, Dacheng Tao, and Chao Xu. "A survey on multi-view learning."
    arXiv preprint, arXiv:1304.5634, 2013.

.. [#2zhao_2017] Jing Zhao, Xijiong Xie, Xin Xu, and Shiliang Sun. "Multi-view learning overview: Recent progress and new challenges."
    Information Fusion, 38:43 – 54, 2017.

.. [#3sun_2013] Shiliang Sun. "A survey of multi-view machine learning." Neural Computing and Applications, 23(7-8):2031–2038, 2013.

.. [#4hardoon_2004] David R Hardoon, Sandor Szedmak, and John Shawe-Taylor. "Canonical correlation analysis:An overview with application to learning methods."
    Neural Computation, 16(12):2639–2664, 2004.

.. [#5chao_2017] Guoqing Chao, Shiliang Sun, and J. Bi. "A survey on multi-view clustering."
    arXiv preprint, arXiv:1712.06246, 2017.

.. [#6yang_2014] Yuhao Yang, Chao Lan, Xiaoli Li, Bo Luo, and Jun Huan. "Automatic social circle detectionusing multi-view clustering."
    In Proceedings of the 23rd ACM International Conferenceon Conference on Information and Knowledge Management, pages 1019–1028, 2014.
