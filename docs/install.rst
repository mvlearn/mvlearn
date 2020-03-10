Install
=======

Below we assume you have the default Python environment already configured on
your computer and you intend to install ``mvlearn`` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip`` (the Python package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install the released version
----------------------------

Install the current release of ``mvlearn`` with ``pip``::

    $ pip install mvlearn

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade mvlearn

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip install --user mvlearn

Alternatively, you can manually download ``mvlearn`` from
`GitHub <https://github.com/NeuroDataDesign/mvlearn>`_  or
`PyPI <https://pypi.org/project/mvlearn/>`_.
To install one of these versions, unpack it and run the following from the
top-level source directory using the Terminal::

    $ pip install .

Python package dependencies
---------------------------
mvlearn requires the following packages:

- graspy
- matplotlib
- numpy
- pillow
- pandas
- scikit-learn
- scipy
- seaborn
- torch
- torchvision
- tqdm

Hardware requirements
---------------------
The `mvlearn` package requires only a standard computer with enough RAM to support the in-memory operations and free memory to install required packages. 

OS Requirements
---------------
This package is supported for *Linux* and *macOS* and can also be run on Windows machines.


Testing
-------
mvlearn uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.