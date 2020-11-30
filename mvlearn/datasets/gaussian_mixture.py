# License: MIT
# Author: Ronan Perry

import numpy as np
from scipy.stats import ortho_group


def make_gaussian_mixture(
    n_samples,
    centers,
    covariances,
    transform='linear',
    noise=None,
    noise_dims=None,
    class_probs=None,
    random_state=None,
    shuffle=False,
    shuffle_random_state=None,
    seed=1,
    return_latents=False,
):
    r"""
    Creates a two-view dataset from a Gaussian mixture model and
    a transformation.

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
    transform : 'linear' | 'sin' | poly' | callable, (default 'linear')
        Transformation to perform on the latent variable. If a function,
        applies it to the latent. Otherwise uses an implemented function.
    noise : double or None (default=None)
        Variance of mean zero Gaussian noise added to the first view
    noise_dims : int or None (default=None)
        Number of additional dimensions of standard normal noise to add
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
    return_latents : boolean (defaul False)
        If true, returns the non-noisy latent variables

    Returns
    -------
    Xs : list of np.ndarray, each shape (n_samples, n_features)
        The latent data and its noisy transformation

    y : np.ndarray, shape (n_samples,)
        The integer labels for each sample's Gaussian membership

    latents : np.ndarray, shape (n_samples, n_features)
        The non-noisy latent variables. Only returned if
        ``return_latents=True``.

    Notes
    -----
    For each class :math:`i` with prior probability :math:`p_i`,
    center and covariance matrix :math:`\mu_i` and :math:`\Sigma_i`, and
    :math:`n` total samples, the latent data is sampled such that:

    .. math::
        (X_1, y_1), \dots, (X_{np_i}, Y_{np_i}) \overset{i.i.d.}{\sim}
            \mathcal{N}(\mu_i, \Sigma_i)

    Two views of data are returned, the first being the latent samples and
    the second being a specified transformation of the latent samples.
    Additional noise may be added to the first view or added as noise
    dimensions to both views.

    Examples
    --------
    >>> from mvlearn.datasets import GaussianMixture
    >>> import numpy as np
    >>> n_samples = 10
    >>> centers = [[0,1], [0,-1]]
    >>> covariances = [np.eye(2), np.eye(2)]
    >>> Xs, y = GaussianMixture(n_samples, centers, covariances,
    ...                         shuffle=True, shuffle_random_state=42)
    >>> print(y)
    [1. 0. 1. 0. 1. 0. 1. 0. 0. 1.]
    """
    centers = np.asarray(centers)
    covariances = np.asarray(covariances)

    if centers.ndim == 1:
        centers = centers[np.newaxis, :]
    if covariances.ndim == 2:
        covariances = covariances[np.newaxis, :]
    if not centers.ndim == 2:
        msg = "centers is of the incorrect shape"
        raise ValueError(msg)
    if not covariances.ndim == 3:
        msg = "covariance matrix is of the incorrect shape"
        raise ValueError(msg)
    if centers.shape[0] != covariances.shape[0]:
        msg = "The first dimensions of 2D centers and 3D covariances" + \
            "must be equal"
        raise ValueError(msg)
    if centers.dtype == np.dtype(
        "O"
    ) or covariances.dtype == np.dtype("O"):
        msg = "elements of covariances or centers are of " + \
            "inconsistent lengths or are not floats nor ints"
        raise ValueError(msg)
    if class_probs is None:
        class_probs = np.ones(centers.shape[0])
        class_probs /= centers.shape[0]
    elif sum(class_probs) != 1.0:
        msg = "elements of `class_probs` must sum to 1"
        raise ValueError(msg)
    if len(centers) != len(class_probs) or len(
        covariances
    ) != len(class_probs):
        msg = (
            "centers, covariances, and class_probs must be of equal length"
        )
        raise ValueError(msg)

    np.random.seed(random_state)
    latent = np.concatenate(
        [
            np.random.multivariate_normal(
                centers[i],
                covariances[i],
                size=int(class_probs[i] * n_samples),
            )
            for i in range(len(class_probs))
        ]
    )
    y = np.concatenate(
        [
            i * np.ones(int(class_probs[i] * n_samples))
            for i in range(len(class_probs))
        ]
    )

    # shuffle latent samples and labels
    if shuffle:
        np.random.seed(shuffle_random_state)
        indices = np.arange(latent.shape[0]).squeeze()
        np.random.shuffle(indices)
        latent = latent[indices, :]
        y = y[indices]

    if callable(transform):
        X = np.asarray([transform(x) for x in latent])
    elif not type(transform) == str:
        raise TypeError(
            "'transform' must be of type string or a callable function," +
            f"not {type(transform)}"
        )
    elif transform == "linear":
        X = _linear2view(latent)
    elif transform == "poly":
        X = _poly2view(latent)
    elif transform == "sin":
        X = _sin2view(latent)
    else:
        raise ValueError(
            "Transform type must be one of {'linear', 'poly', 'sin'}" +
            f" or a callable function. Not {transform}"
        )

    if noise is not None:
        Xs = [latent + np.sqrt(noise) * np.random.randn(*latent.shape), X]
    else:
        Xs = [latent, X]

    # if random_state is not None, make sure both views are independent
    # but reproducible
    if noise_dims is not None:
        np.random.seed(random_state)
        Xs = [_add_noise(X, noise_dims) for X in Xs]

    if return_latents:
        return Xs, y, latent
    else:
        return Xs, y


def _add_noise(X, n_noise):
    """Appends dimensions of standard normal noise to X
    """
    noise_vars = np.random.randn(X.shape[0], n_noise)
    return np.hstack((X, noise_vars))


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
