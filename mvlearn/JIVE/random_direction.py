import numpy as np
from jive.utils import svd_wrapper
from sklearn.externals.joblib import Parallel, delayed


def sample_randdir(num_obs, signal_ranks, R=1000, n_jobs=None):
    """
    Draws samples for the random direction bound.

    Parameters
    ----------

    num_obs: int
        Number of observations.

    signal_ranks: list of ints
        The initial signal ranks for each block.

    R: int
        Number of samples to draw.

    n_jobs: int, None
        Number of jobs for parallel processing using
        sklearn.externals.joblib.Parallel. If None, will not use parallel
        processing.

    Output
    ------
    random_sv_samples: np.array, shape (R, )
        The samples.
    """

    if n_jobs is not None:
        random_sv_samples = Parallel(n_jobs=n_jobs)\
            (delayed(_get_sample)(num_obs, signal_ranks)
             for i in range(R))

    else:
        random_sv_samples = [_get_sample(num_obs, signal_ranks)
                             for r in range(R)]

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
