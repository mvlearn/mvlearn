..  -*- coding: utf-8 -*-

.. _contents:

Overview of mvlearn_
===================

.. _mvlearn: https://github.com/NeuroDataDesign/mvlearn

mvlearn is a Python module for multiview learning. 

Motivation
----------

In many data sets, there are multiple measurement modalities of the same subject, i.e. multiple *X* matrices (views) for the same class label vector *y*. For example, a set of diseased and healthy patients in a neuroimaging study may undergo both CT and MRI scans. Traditional methods for inference and analysis are often poorly suited to account for multiple views of the same subject as they cannot account for complementing views that hold different statistical properties. While single-view methods are consolidated in well-documented packages such as scikit-learn, there is no equivalent for multi-view methods. In this package, we a provide well-documented and tested collection of utilities and algorithms designed for the processing and analysis of multiview data sets.

Python
------

Python is a powerful programming language that allows concise expressions of network
algorithms.  Python has a vibrant and growing ecosystem of packages that
mvlearn uses to provide more features such as numerical linear algebra. In order to make the most out of mvlearn you will want to know how
to write basic programs in Python.  Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_.

Free software
-------------

mvlearn is free software; you can redistribute it and/or modify it under the
terms of the :doc:`Apache-2.0 </license>`.  We welcome contributions.
Join us on `GitHub <https://github.com/NeuroDataDesign/mvlearn>`_.

History
-------

mvlearn was developed during the end of 2019 by Richard Guo, Ronan Perry, Gavin Mischler, Theo Lee, Alexander Chang, Arman Koul, and Cameron Franz, a team out of the Johns Hopkins University NeuroData group.

Documentation
=============

mvearn is a python package of multiview learning tools.

.. toctree::
   :maxdepth: 1

   install
   tutorial
   reference/index
   contributing
   news
   license

.. toctree::
   :maxdepth: 1
   :caption: Useful Links

   GraSPy @ GitHub <http://www.github.com/neurodata/graspy/>
   GraSPy @ PyPI <https://pypi.org/project/graspy/>
   Issue Tracker <https://github.com/neurodata/graspy/issues>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`