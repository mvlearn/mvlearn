# mvlearn

[![Python](https://img.shields.io/badge/python-3.7-blue.svg)]()
[![Build Status](https://travis-ci.com/NeuroDataDesign/mvlearn.svg?branch=master)](https://travis-ci.com/NeuroDataDesign/mvlearn)
[![Documentation Status](https://readthedocs.org/projects/mvlearn/badge/?version=latest)](https://mvlearn.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/NeuroDataDesign/mvlearn/branch/master/graph/badge.svg)](https://codecov.io/gh/NeuroDataDesign/mvlearn)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

`mvlearn` is an open-source Python software packaged for multiview learning learning tools.

- [**Installation Guide**](https://mvlearn.netlify.com/install.html)
- [**Documentation**](https://mvlearn.netlify.com/index.html)
- [**Tutorials**](https://mvlearn.netlify.com/tutorials.html)
- [**Source Code**](https://github.com/NeuroDataDesign/mvlearn/tree/master/mvlearn)
- [**Issues**](https://github.com/NeuroDataDesign/mvlearn/issues)
- [**Contribution Guide**](https://mvlearn.netlify.com/contributing.html)

In many data sets, there are multiple measurement modalities of the same subject, i.e. multiple *X* matrices (views) for the same class label vector *y*. For example, one may have both an MRI and CT scan of a diseased patient. Traditional methods for inference and analysis are often poorly suited to account for multiple views of the same subject as they cannot account for complementing views that hold different statistical properties. `mvlearn` provides a well-documented and tested collection of utilities and algorithms designed for the processing and analysis of multiview data sets.
