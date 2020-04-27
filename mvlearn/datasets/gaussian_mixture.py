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

import numpy as np
from scipy.stats import ortho_group
import collections


def _add_noise(X, n_noise, random_state=None):
    """Appends dimensions of standard normal noise to X
    """
    np.random.seed(random_state)
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
    def __init__(self, mu, sigma, n, class_probs=None, random_state=None,
                 shuffle=False, shuffle_random_state=None):
        r"""
        Creates a GaussianMixture object and samples a latent variable from a
        multivariate Gaussian distribution according to the specified
        parameters and class probability priors if set.

        For each class :math:`i` with prior probability :math:`p_i`,
        mean and variance :math:`\mu_i` and :math:`\sigma^2_i`, and :math:`n`
        total samples, the latent data is sampled such that:

        .. math::
            (X_1, y_1), \dots, (X_{np_i}, Y_{np_i}) \overset{i.i.d.}{\sim}
                \mathcal{N}(\mu_i, \sigma^2_i)

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
            whose entries sum to 1. If `None`, then data is sampled from one
            class.
        random_state : int, default=None
            If set, can be used to reproduce the data generated.
        shuffle : bool, default=False
            If ``True``, data is shuffled so the labels are not ordered.
        shuffle_random_state : int, default=None
            If given, then sets the random state for shuffling the samples.
            Ignored if ``shuffle=False``.

        Attributes
        ----------
        latent : np.ndarray
            Latent distribution data. latent[:,i] is randomly sampled from
            a gaussian distribution with mean mu[i] and covariance sigma[i].
        Xs : list of array-like
            List of views of data created by transforming the latent.
        mu : np.ndarray
            Means of gaussian blobs in latent distribution. mu[:,i] is
            mean of *ith* gaussian.
        sigma : np.ndarray
            Covariance matrices of gaussian blobs in the latent distribution.
            sigma[i] is the covariance matrix of the *ith* gaussian.
        class_probs : array-like, default=None
            A list correponding to the fraction of samples from each class and
            whose entries sum to 1. If `None`, then data is sampled from one
            class.
        views : int
            Number of views in the multi-view gaussian mixture.
        random_state : int
            Random state for data generation.
        shuffle : bool
            Whether or not to shuffle data when creating.
        shuffle_random_state : int
            Random state for data shuffling.

        Examples
        --------
        >>> from mvlearn.datasets import GaussianMixture
        >>> import numpy as np
        >>> n = 10
        >>> mu = [[0,1], [0,-1]]
        >>> sigma = [np.eye(2), np.eye(2)]
        >>> class_probs = [0.5, 0.5]
        >>> GM = GaussianMixture(mu,sigma,n,class_probs=class_probs,
        ...                      shuffle=True, shuffle_random_state=42)
        >>> GM = GM.sample_views(transform='poly', n_noise=2)
        >>> Xs, y = GM.get_Xy()
        >>> print(y)
        [1. 0. 1. 0. 1. 0. 1. 0. 0. 1.]
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
        self.random_state = random_state
        self.shuffle = shuffle
        self.shuffle_random_state = shuffle_random_state

        if class_probs is None:
            if self.mu.ndim > 1 or self.sigma.ndim > 2:
                msg = "mu or sigma of the wrong size"
                raise ValueError(msg)
            np.random.seed(random_state)
            self.latent = np.random.multivariate_normal(mu, sigma, size=n)
            self.y = None
        else:
            if len(self.mu) != len(class_probs) or len(self.sigma) != len(
                    class_probs):
                msg = "mu, sigma, and class_probs must be of equal length"
                raise ValueError(msg)
            np.random.seed(random_state)
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

        # shuffle latent samples and labels
        if self.shuffle:
            np.random.seed(self.shuffle_random_state)
            indices = np.arange(self.latent.shape[0]).squeeze()
            np.random.shuffle(indices)
            self.latent = self.latent[indices, :]
            self.y = self.y[indices]

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
        self.Xs = [_add_noise(X, n_noise=n_noise,
                              random_state=self.random_state)
                   for X in self.Xs]

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
