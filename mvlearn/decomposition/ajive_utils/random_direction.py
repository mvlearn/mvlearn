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
from .utils import svd_wrapper


def sample_randdir(num_obs, signal_ranks, R=1000):
    r"""
    Draws samples for the random direction bound.

    Parameters
    ----------

    num_obs: int
        Number of observations.

    signal_ranks: list of ints
        The initial signal ranks for each block.

    R: int
        Number of samples to draw.

    Returns
    -------
    random_sv_samples: np.array
        - random_sv_samples shape = (R, )
        The samples
    """

    random_sv_samples = [
        _get_sample(num_obs, signal_ranks) for r in range(R)
        ]

    return np.array(random_sv_samples)


def _get_sample(num_obs, signal_ranks):
    M = [None for _ in range(len(signal_ranks))]
    for k in range(len(signal_ranks)):

        # sample random orthonormal basis
        Z = np.random.normal(size=(num_obs, signal_ranks[k]))
        M[k] = np.linalg.qr(Z)[0]

    # compute largest sing val of random joint matrix
    M = np.bmat(M)
    _, svs, __ = svd_wrapper(M, rank=1)

    return max(svs) ** 2
