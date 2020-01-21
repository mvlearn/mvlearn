# mvlearn

[![Python](https://img.shields.io/badge/python-3.7-blue.svg)]()
[![Build Status](https://travis-ci.com/NeuroDataDesign/mvlearn.svg?branch=master)](https://travis-ci.com/NeuroDataDesign/mvlearn)
[![Documentation Status](https://readthedocs.org/projects/mvlearn/badge/?version=latest)](https://mvlearn.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/NeuroDataDesign/mvlearn/branch/master/graph/badge.svg)](https://codecov.io/gh/NeuroDataDesign/mvlearn)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


mvlearn is a Python module for multiview learning. 

## Summary
In many data sets, there are multiple measurement modalities of the same subject, i.e. multiple *X* matrices (views) for the same class label vector *y*. For example, a set of diseased and healthy patients in a neuroimaging study may undergo both CT and MRI scans. Traditional methods for inference and analysis are often poorly suited to account for multiple views of the same subject as they cannot account for complementing views that hold different statistical properties. While single-view methods are consolidated in well-documented packages such as scikit-learn, there is no equivalent for multi-view methods. In this package, we a provide well-documented and tested collection of utilities and algorithms designed for the processing and analysis of multiview data sets.

## Tutorials
Tutorials can be found in the docs [folder](https://github.com/NeuroDataDesign/mvlearn/tree/master/docs/tutorials)

## Installation

### Install from pip

```shell
pip3 install mvlearn
```

### Install from Github

```
git clone https://github.com/NeuroDataDesign/mvlearn.git
cd mvlearn
python3 setup.py install
```

### Requirements
This package is written for Python3, currently supported for Python 3.6 and 3.7.

## Contributing
We welcome contributions from anyone. Please see our [contribution guidelines](https://github.com/NeuroDataDesign/mvlearn/blob/master/Contributing.md) before making a pull request. Our 
[issues](https://github.com/NeuroDataDesign/mvlearn/issues) page is full of places we could use help! 
If you have an idea for an improvement not listed there, please 
[make an issue](https://github.com/NeuroDataDesign/mvlearn/issues/new) first so you can discuss with the 
developers. 

## License
This project is covered under the [Apache 2.0 License](https://github.com/NeuroDataDesign/mvlearn/blob/master/LICENSE).
