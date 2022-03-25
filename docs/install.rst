Install
=======

``mvlearn`` can be installed by using ``pip``, GitHub, or through the conda-forge
channel into an existing ``conda`` environment.

**IMPORTANT NOTE:** ``mvlearn`` has an optional dependencies for certain functions,
and so special instructions must be followed to include these
optional dependencies in the installation (if you do not have those packages already)
in order to access all the features within ``mvlearn``.
More details can be found in :ref:`extraDependencyAnchor`.

Installing the released version with pip
----------------------------------------

Below we assume you have the default Python3 environment already configured on
your computer and you intend to install ``mvlearn`` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip3`` (the Python3 package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip3`` first.

Install the current release of ``mvlearn`` with ``pip3``::

    $ pip3 install mvlearn

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip3 install --upgrade mvlearn

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip3 install --user mvlearn

Alternatively, you can manually download ``mvlearn`` from
`GitHub <https://github.com/mvlearn/mvlearn>`_  or
`PyPI <https://pypi.org/project/mvlearn/>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip3 install -e .

This will install ``mvlearn`` and the required dependencies (see below).

.. _extraDependencyAnchor:

Including optional dependencies for full functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A small subset of functions require specific extra dependencies not installed
by default with the core installation. Each bullet point denotes a collection
of functions, with corresponding keyword enclosed in the brackets [].

* [torch]: ``DCCA``, ``SplitAE``
* [multiviewica]: ``MultiviewICA``, ``GroupICA``

If you wish to use these functions, you must install their required
dependencies. These are listed in the package requirements folder
with corresponding keyword names
for manual installation or can be installed from PyPI by simply calling::

    $ pip3 install mvlearn[keyword]

where 'keyword' is from the list above, bracketed.
To upgrade the package and torch requirements::

    $ pip3 install --upgrade mvlearn[keyword]

If you have the package locally, from the top level folder call::

    $ pip3 install -e .[keyword]

.. _condaAnchor:

Installing the released version with conda-forge
------------------------------------------------

Here, we assume you have created a conda environment with one of the
accepted python versions, and you intend to install the full ``mvlearn``
release into it (with extra dependencies included). For more information
about using conda-forge feedstocks, see the `about page <https://conda-forge.org/>`_,
or the `mvlearn feedstock <https://github.com/conda-forge/mvlearn-feedstock>`_.

To install ``mvlearn`` with conda, run::

	$ conda install -c conda-forge mvlearn

To list all versions of ``mvlearn`` available on your platform, use::

	$ conda search mvlearn --channel conda-forge


Python package dependencies
---------------------------
``mvlearn`` requires the following packages:

- matplotlib >=3.0.0
- numpy >=1.17.0
- scikit-learn >=0.19.1
- scipy >=1.1.0
- seaborn >=0.9.0
- joblib >=0.11

with optional [torch] dependencies,

- torch >=1.1.0
- tqdm

optional [multiviewica] dependencies,

- python-picard >=0.4
- multiviewica >=0.0.1

and optional [graspy] dependencies,

- graspologic >=1.0.0

Currently, ``mvlearn`` is supported for Python 3.6, 3.7, and 3.8.

Hardware requirements
---------------------
The ``mvlearn`` package requires only a standard computer with enough RAM to support the in-memory operations and free memory to install required packages. 

OS Requirements
---------------
This package is supported for *Linux* and *macOS* and can also be run on Windows machines.

Testing
-------
``mvlearn`` uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.