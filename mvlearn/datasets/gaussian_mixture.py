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
    def __init__(
        self,
        n_samples,
        centers,
        covariances,
        class_probs=None,
        random_state=None,
        shuffle=False,
        shuffle_random_state=None,
        seed=1,
    ):
        r"""
        Creates an object with a fixed latent variable sampled from a
        (potentially) multivariate Gaussian distribution according to the
        specified parameters and class probability priors.

        Parameters
        ----------
        n_samples : int
            The number of points in each view, divided across Gaussians per
            `class_probs`.
        centers : 1D array-like or list of 1D array-likes
            The mean(s) of the Gaussian(s) from which the latent
            points are sampled. If is a list of 1D array-likes, each is the
            mean of a distinct Gaussian, sampled from with
            probability given by `class_probs`. Otherwise is the mean of a
            single Gaussian from which all are sampled.
        covariances : 2D array-like or list of 2D array-likes
            The covariance matrix(s) of the Gaussian(s), matched
            to the specified centers.
        class_probs : array-like, default=None
            A list of probabilities specifying the probability of a latent
            point being sampled from each of the Gaussians. Must sum to 1. If
            None, then is taken to be uniform over the Gaussians.
        random_state : int, default=None
            If set, can be used to reproduce the data generated.
        shuffle : bool, default=False
            If ``True``, data is shuffled so the labels are not ordered.
        shuffle_random_state : int, default=None
            If given, then sets the random state for shuffling the samples.
            Ignored if ``shuffle=False``.

        Attributes
        ----------
        latent_ : np.ndarray, of shape (n_samples, n_dims)
            Latent distribution data. latent[i] is randomly sampled from
            a gaussian distribution with mean centers[i] and covariance
            covariances[i].
        y_ : np.ndarray, of shape (n_samples)
        Xs_ : list of array-like, of shape (2, n_samples, n_dims)
            List of views of data created by transforming the latent.
        centers : ndarray of shape (n_classes, n_dims)
            The mean(s) of the Gaussian(s) from which the latent
            points are sampled.
        covariances : ndarray of shape (n_classes, n_dims, n_dims)
            The covariance matrix(s) of the Gaussian(s).
        class_probs_ : array-like of shape (n_classes,)
            A list correponding to the fraction of samples from each class and
            whose entries sum to 1.

        Notes
        -----
        For each class :math:`i` with prior probability :math:`p_i`,
        center and covariance matrix :math:`\mu_i` and :math:`\Sigma_i`, and
        :math:`n` total samples, the latent data is sampled such that:

        .. math::
            (X_1, y_1), \dots, (X_{np_i}, Y_{np_i}) \overset{i.i.d.}{\sim}
                \mathcal{N}(\mu_i, \Sigma_i)

        Examples
        --------
        >>> from mvlearn.datasets import GaussianMixture
        >>> import numpy as np
        >>> n_samples = 10
        >>> centers = [[0,1], [0,-1]]
        >>> covariances = [np.eye(2), np.eye(2)]
        >>> GM = GaussianMixture(n_samples, centers, covariances,
        ...                      shuffle=True, shuffle_random_state=42)
        >>> GM = GM.sample_views(transform='poly', n_noise=2)
        >>> Xs, y = GM.get_Xy()
        >>> print(y)
        [1. 0. 1. 0. 1. 0. 1. 0. 0. 1.]
        """
        self.centers_ = np.asarray(centers)
        self.covariances_ = np.asarray(covariances)
        self.n_samples = n_samples
        self.class_probs_ = class_probs
        self.random_state = random_state
        self.shuffle = shuffle
        self.shuffle_random_state = shuffle_random_state

        if self.centers_.ndim == 1:
            self.centers_ = self.centers_[np.newaxis, :]
        if self.covariances_.ndim == 2:
            self.covariances_ = self.covariances_[np.newaxis, :]
        if not self.centers_.ndim == 2:
            msg = "centers is of the incorrect shape"
            raise ValueError(msg)
        if not self.covariances_.ndim == 3:
            msg = "covariances if of the incorrect shape"
            raise ValueError(msg)
        if self.centers_.shape[0] != self.covariances_.shape[0]:
            msg = "The first dimensions of 2D centers and 3D covariances \
                must be equal"
            raise ValueError(msg)
        if self.centers_.dtype == np.dtype(
            "O"
        ) or self.covariances_.dtype == np.dtype("O"):
            msg = "elements of covariances or centers are of \
                inconsistent lengths or are not floats nor ints"
            raise ValueError(msg)
        if self.class_probs_ is None:
            self.class_probs_ = np.ones(self.centers_.shape[0])
            self.class_probs_ /= self.centers_.shape[0]
        elif sum(self.class_probs_) != 1.0:
            msg = "elements of `class_probs` must sum to 1"
            raise ValueError(msg)
        if len(self.centers_) != len(self.class_probs_) or len(
            self.covariances_
        ) != len(self.class_probs_):
            msg = (
                "centers, covariances, and class_probs must be of equal length"
            )
            raise ValueError(msg)

        np.random.seed(self.random_state)
        self.latent_ = np.concatenate(
            [
                np.random.multivariate_normal(
                    self.centers_[i],
                    self.covariances_[i],
                    size=int(self.class_probs_[i] * self.n_samples),
                )
                for i in range(len(self.class_probs_))
            ]
        )
        self.y_ = np.concatenate(
            [
                i * np.ones(int(self.class_probs_[i] * self.n_samples))
                for i in range(len(self.class_probs_))
            ]
        )

        # shuffle latent samples and labels
        if self.shuffle:
            np.random.seed(self.shuffle_random_state)
            indices = np.arange(self.latent_.shape[0]).squeeze()
            np.random.shuffle(indices)
            self.latent_ = self.latent_[indices, :]
            self.y_ = self.y_[indices]

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
            X = np.asarray([transform(x) for x in self.latent_])
        elif not type(transform) == str:
            raise TypeError(
                f"'transform' must be of type string or a callable function,\
                not {type(transform)}"
            )
        elif transform == "linear":
            X = _linear2view(self.latent_)
        elif transform == "poly":
            X = _poly2view(self.latent_)
        elif transform == "sin":
            X = _sin2view(self.latent_)
        else:
            raise ValueError(
                "Transform type must be one of {'linear', 'poly'\
                , 'sin'} or a callable function. Not "
                + f"{transform}"
            )

        self.Xs_ = [self.latent_, X]

        # if random_state is not None, make sure both views are independent
        # but reproducible
        if self.random_state is None:
            self.Xs_ = [
                _add_noise(X, n_noise=n_noise, random_state=self.random_state)
                for X in self.Xs_
            ]
        else:
            self.Xs_ = [
                _add_noise(
                    X, n_noise=n_noise, random_state=(self.random_state + i)
                )
                for i, X in enumerate(self.Xs_)
            ]

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
            returns the latent variables instead of Xs.
        """
        if latents:
            return (self.latent_, self.y_)
        else:
            if not hasattr(self, "Xs_"):
                raise NameError("sample_views has not been called yet")
            return (self.Xs_, self.y_)
