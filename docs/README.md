# Building Docs

We currently use Sphinx

If you only want to get the documentation, note that pre-build versions can be found at

[mvlearn.neurodata.io](https://mvlearn.github.io/)

## Instructions

### Python Dependencies

You will need to install all the dependencies as defined in `requirements.txt` file. The following packages are needed:

    sphinx>=4.2.0
    sphinx_rtd_theme>=1.0.0
    ipython>=7.4
    ipykernel>=5.1.0
    numpydoc>=1.1.0
    recommonmark==0.5.0
    sphinx-gallery

The above can be installed by entering:

    pip3 install -r requirements.txt

in the `doc/` directory.

### Pandoc dependency

In addition, you need to install `pandoc` for `nbsphinx`. If you are on linux, you can enter: 

    sudo apt-get install pandoc

If you are on macOS and have `homebrew` installed, you can enter:

    brew install pandoc

Otherwise, you can visit [pandoc installing page](https://pandoc.org/installing.html) for more information.

## Generating the documentation

To build the HTML documentation, enter:

    make html

in the `doc/` directory. If all goes well, this will generate a `_build/html/` subdirectory containing the built documentation.
