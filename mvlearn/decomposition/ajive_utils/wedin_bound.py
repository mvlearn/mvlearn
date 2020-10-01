# MIT License

# Copyright (c) [2017] [Iain Carmichael]

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np


def _get_wedin_samples(X, U, D, V, rank, R=1000):
    r"""
    Computes the wedin bound using the sample-project procedure. This method
    does not require the full SVD.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The data

    U, D, V : array-likes
        The partial SVD of X=UDV^T

    rank : int
        The rank of the signal space

    R : int
        Number of samples for resampling procedure

    Returns
    -------
    wedin_bound_samples : array of resampled wedin bounds
    """

    # resample for U and V
    U_norm_samples = _norms_sample_project(
        X=X.T, basis=U[:, :rank], R=R
    )

    V_norm_samples = _norms_sample_project(
        X=X, basis=V[:, :rank], R=R
    )

    sigma_min = D[rank - 1]  # TODO: double check -1
    wedin_bound_samples = [
        min(max(U_norm_samples[r], V_norm_samples[r]) / sigma_min, 1)
        for r in range(R)
    ]

    return wedin_bound_samples


def _norms_sample_project(X, basis, R=1000):
    r"""
    Samples vectors from space orthogonal to signal space as follows
    - sample random vector from isotropic distribution
    - project onto orthogonal complement of signal space and normalize

    Parameters
    ---------
    X: array-like, shape (N, D)
        The observed data

    B: array-like
        The basis for the signal col/rows space (e.g. the left/right singular\
        vectors)

    rank: int
        Number of columns to resample

    R: int
        Number of samples

    Returns
    -------
    samples : Array of the resampled norms
    """
    samples = [_get_sample(X, basis) for r in range(R)]

    return np.array(samples)


def _get_sample(X, basis):
    """
    Estimates magnitude of noise matrix projected onto signal matrix.
    """
    dim, rank = basis.shape

    # sample from isotropic distribution
    vecs = np.random.normal(size=(dim, rank))

    # project onto space orthogonal to cols of B
    # vecs = (np.eye(dim) - np.dot(basis, basis.T)).dot(vecs)
    vecs = vecs - np.dot(basis, np.dot(basis.T, vecs))

    # orthonormalize
    vecs, _ = np.linalg.qr(vecs)

    # compute operator L2 norm
    return np.linalg.norm(X.dot(vecs), ord=2)
