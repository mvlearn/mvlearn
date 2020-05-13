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
