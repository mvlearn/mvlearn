..  -*- coding: utf-8 -*-

.. _contents:

Overview of mvlearn_
====================

.. _mvlearn: https://github.com/mvlearn/mvlearn

mvlearn is a Python module for multiview learning. 

Motivation
----------

mvlearn aims to serve as a community-driven open-source software package that offers reference implementations for algorithms and methods related to multiview learning (machine learning in settings where there are multiple incommensurate views or feature sets for each sample). It brings together the most widely-used tools in this setting with a standardized scikit-learn like API, well tested code and high-quality documentation. Doing so we aim to facilitate application, extension, comparison of methods, and offer a foundation for research into new multiview algorithms. We welcome new contributors and the addition of methods with proven efficacy and current use.

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

Documentation
=============

mvlearn is a Python package of multiview learning tools.

.. toctree::
   :maxdepth: 2
   :caption: Using mvlearn

   install
   auto_examples/index
   references/index

.. toctree::
   :maxdepth: 2
   :caption: Developer Information

   contributing
   changelog
   license

.. toctree::
   :maxdepth: 1
   :caption: Useful Links

   mvlearn @ GitHub <https://github.com/mvlearn/mvlearn>
   mvlearn @ PyPI <https://pypi.org/project/mvlearn/>
   Issue Tracker <https://github.com/mvlearn/mvlearn/issues>


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
