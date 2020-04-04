# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Original work Copyright (c) 2016 Vahid Noroozi
# Modified work Copyright 2019 Zhanghao Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import numpy as np
from scipy.stats import ortho_group
import collections


def _add_noise(X, n_noise, random_state=None):
    """Appends dimensions of standard normal noise to X
    """
    noise = np.random.randn(X.shape[0], n_noise)
    return np.hstack((X, noise))


def _linear2view(X):
    """Rotates the data, a linear transformation
    """
    if X.shape[1] == 1:
        X = -X
    else:
        np.random.seed(2)
        X = X @ ortho_group.rvs(X.shape[1])
    return X


def _poly2view(X):
    """Applies a degree 2 polynomial transform to the data
    """
    X = np.asarray([np.power(x, 2) for x in X])
    return X


def _sin2view(X):
    """Applies a sinusoidal transformation to the data
    """
    X = np.asarray([np.sin(x) for x in X])
    return X


class GaussianMixture:
    def __init__(self, mu, sigma, n, class_probs=None):
        r"""
        Creates a GaussianMixture object and samples a latent variable from a
        multivariate Gaussian distribution according to the specified
        parameters and class probability priors if set.

        Parameters
        ----------
        mu : 1D array-like or list of 1D array-likes
            The mean(s) of the multivariate Gaussian(s). If `class_probs` is
            None (default), then is a 1D array-like of the multivariate mean.
            Otherwise is a list of means of the same length as `class_probs`.
        sigma : 2D array-like or list of 2D array-likes
            The variance(s) of the multivariate Gaussian(s). If `class_probs`
            is None (default), then is a 2D array-like of the multivariate
            variance. Otherwise is a list of variances of the same length as
            `class_probs`.
        n : int
            The number of samples to sample.
        class_probs : array-like, default=None
            A list correponding to the fraction of samples from each class and
            whose entries sum to 1. If `None`, then data is sampled from a one
            class.
        """
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)

        if self.mu.shape[0] != self.sigma.shape[0]:
            msg = "length of mu and sigma must be equal"
            raise ValueError(msg)
        if (self.mu.dtype == np.dtype("O") or 
                self.sigma.dtype == np.dtype("O")):
            msg = "elements of sigma or mu are of inconsistent lengths or \
                are not floats nor ints"
            raise ValueError(msg)
        if class_probs is not None and sum(class_probs) != 1.0:
            msg = "elements of `class_probs` must sum to 1"
            raise ValueError(msg)

        self.class_probs = class_probs
        self.views = len(mu)

        if class_probs is None:
            if self.mu.ndim > 1 or self.sigma.ndim > 2:
                msg = "mu or sigma of the wrong size"
                raise ValueError(msg)
            self.latent = np.random.multivariate_normal(mu, sigma, size=n)
            self.y = None
        else:
            if len(self.mu) != len(class_probs) or len(self.sigma) != len(
                    class_probs):
                msg = "mu, sigma, and class_probs must be of equal length"
                raise ValueError(msg)
            self.latent = np.concatenate(
                [
                    np.random.multivariate_normal(
                        self.mu[i], self.sigma[i], size=int(class_probs[i] * n)
                    )
                    for i in range(len(class_probs))
                ]
            )
            self.y = np.concatenate(
                [
                    i * np.ones(int(class_probs[i] * n))
                    for i in range(len(class_probs))
                ]
            )

    def sample_views(self, transform="linear", n_noise=1):
        r"""
        Transforms one latent view by specified transformation and adds noise.

        Parameters
        ----------
        transform : function or one of {'linear', 'sin', poly'},
            default = 'linear'
            Transformation to perform on the latent variable. If a function,
            applies it to the latent. Otherwise uses an implemented function.
        n_noise : int, default = 1
            number of noise dimensions to add to transformed latent

        Returns
        -------
        self : returns an instance of self
        """

        if callable(transform):
            X = np.asarray([transform(x) for x in self.latent])
        elif not type(transform) == str:
            raise TypeError(
                f"'transform' must be of type string or a callable function,\
                not {type(transform)}"
            )
        elif transform == "linear":
            X = _linear2view(self.latent)
        elif transform == "poly":
            X = _poly2view(self.latent)
        elif transform == "sin":
            X = _sin2view(self.latent)
        else:
            raise ValueError(
                "Transform type must be one of {'linear', 'poly'\
                , 'sin'} or a callable function. Not "
                + f"{transform}"
            )

        self.Xs = [self.latent, X]
        self.Xs = [_add_noise(X, n_noise=n_noise) for X in self.Xs]

        return self

    def get_Xy(self, latents=False):
        r"""
        Returns the sampled views or latent variables.

        Parameters
        ----------
        latents : boolean, default=False
            If true, returns the latent variables rather than the
            transformed views.

        Returns
        -------
        (Xs, y) : the transformed views and their labels. If `latents=True`,
            returns the latent variables instead.
        """
        if latents:
            return (self.latent, self.y)
        else:
            if not hasattr(self, "Xs"):
                raise NameError("sample_views has not been called yet")
            return (self.Xs, self.y)
