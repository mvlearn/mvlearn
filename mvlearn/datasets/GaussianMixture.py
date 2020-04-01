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


def add_noise(X, n_noise, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    noise = np.random.randn(X.shape[0], n_noise)
    return np.hstack((X, noise))


def linear2view(X):
    if X.shape[1] == 1:
        X = -X
    else:
        np.random.seed(2)
        X = X @ ortho_group.rvs(X.shape[1])
    return X


def poly2view(X):
    X = np.asarray([np.power(x, 2) for x in X])
    return X


def polyinv2view(X):
    X = np.asarray([np.cbrt(x) for x in X])
    return X


def sin2view(X):
    X = np.asarray([np.sin(x) for x in X])
    return X


class GaussianMixture:
    def __init__(self, n, mu, sigma, class_probs=None):
        r"""

        """
        self.mu = mu
        self.sigma = sigma
        self.views = len(mu)
        self.class_probs = class_probs

        if class_probs is None:
            self.latent = np.random.multivariate_normal(mu, sigma, size=n)
            self.y = None
        else:
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
        """
        Transforms one latent view by a specified transformation and adds noise.
    
        Parameters
        ----------
        transform : string, default = 'linear'
            Type of transformation to form views
            - value can be 'linear', 'sin', 'polyinv', 'poly', or 
              custom function
        n_noise : int, default = 1
            number of noise dimensions to add to transformed latent

        Returns
        -------
        self : returns an instance of self
        """

        if callable(transform):
            X = np.asarray([transform(x) for x in self.latent])
        elif not type(transform) == str:
            raise ValueError(
                f"'transform' must be of type string or a callable function,\
                not {type(transform)}"
            )
        elif transform == "linear":
            X = linear2view(self.latent)
        elif transform == "poly":
            X = poly2view(self.latent)
        elif transform == "polyinv":
            X = polyinv2view(self.latent)
        elif transform == "sin":
            X = sin2view(self.latent)

        self.Xs = [self.latent, X]
        self.Xs = [add_noise(X, n_noise=n_noise) for X in self.Xs]

        return self

    def get_Xy(self, latents=False):
        r"""

        """
        if latents:
            return (self.latent, self.y)
        else:
            if not hasattr(self, "Xs"):
                raise NameError("sample_views has not been called yet")
            return (self.Xs, self.y)
