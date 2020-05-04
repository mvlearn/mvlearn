Changelog
=========

Version 0.1.0
-------------

We’re happy to announce the first major stable version of ``mvlearn``.
This version includes multiple new algorithms, more utility functions, as well as significant enhancements to the documentation. Here are some highlights of the big updates.
    - Deep CCA
    - Updated Kernel CCA with multiple kernels
    - Synthetic multi-view dataset generator class, GaussianMixture
    - A new module, ``plotting``, which includes functions for visualizing multi-view data
        * crossviews_plot
        * quick_visualize
    - More detailed tutorial notebooks for all algorithms

Additionally, mvlearn now makes the ``torch`` dependency optional, so users who don’t need the DCCA or SplitAE functionality do not have to import such a large package. To install the full version of mvlearn with ``torch`` from pip, you must include the optional torch in brackets:

    .. code-block:: python
        
        pip3 install mvlearn[torch]

or

    .. code-block:: python
        
        pip3 install --upgrade mvlearn[torch]


To install without `torch`, do:

    .. code-block:: python
        
        pip3 install mvlearn

or

    .. code-block:: python
        
        pip3 install --upgrade mvlearn