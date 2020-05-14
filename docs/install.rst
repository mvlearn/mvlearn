Install
=======

``mvlearn`` can be installed by using ``pip``, GitHub, or through the conda-forge
channel into an existing ``conda`` environment.

**IMPORTANT NOTE:** ``mvlearn`` has an optional dependency to ``torch``
and ``tqdm``, so special instructions must be followed to include these
optional dependencies in the installation (if you do not have those packages already)
in order to access all the features within ``mvlearn``.
More details can be found in :ref:`torchDependencyAnchor`.

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
`GitHub <https://github.com/neurodata/mvlearn>`_  or
`PyPI <https://pypi.org/project/mvlearn/>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip3 install -e .

This will install ``mvlearn`` and the required dependencies (see below).

.. _torchDependencyAnchor:

Including optional torch dependencies for full functionality
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Due to the size of the ``torch`` dependency, it is an optional installation.
Because it, and ``tqdm``, are only used by Deep CCA and SplitAE, they are not
included in the basic ``mvlearn`` download.
If you wish to use functionality associated with these dependencies (Deep CCA
and SplitAE), you must install additional dependencies. You can install
them independently, or to install everything from PyPI, simply call::

    $ pip3 install mvlearn[torch]

To upgrade the package and torch requirements::

    $ pip3 install --upgrade mvlearn[torch]

If you have the package locally, from the top level folder call::

    $ pip3 install -e .[torch]

.. _condaAnchor:

Installing the released version with conda-forge
------------------------------------------------

Here, we assume you have created a conda environment with one of the
accepted python versions, and you intend to install the full ``mvlearn``
release into it (with torch dependencies included). For more information
about using conda-forge feedstocks, see the `about page <https://conda-forge.org/>`_,
or the `mvlearn feedstock <https://github.com/conda-forge/mvlearn-feedstock>`_.

To install ``mvlearn`` with conda, run::

	$ conda install -c conda-forge mvlearn

To list all versions of ``mvlearn`` available on your platform, use::

	$ conda search mvlearn --channel conda-forge


Python package dependencies
---------------------------
``mvlearn`` requires the following packages:

- graspy >=0.1.1
- matplotlib >=3.0.0
- numpy >=1.17.0
- pandas >=0.25.0
- scikit-learn >=0.19.1
- scipy >=1.1.0
- seaborn >=0.9.0
- joblib >=0.11

with optional dependencies

- torch >=1.1.0
- tqdm

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