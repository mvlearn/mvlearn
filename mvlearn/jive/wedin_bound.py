import numpy as np
from sklearn.externals.joblib import Parallel, delayed


def get_wedin_samples(X, U, D, V, rank, R=1000, n_jobs=None):
    """
    Computes the wedin bound using the sample-project procedure. This method
    does not require the full SVD.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The data block.

    U, D, V:
        The partial SVD of X

    rank: int
        The rank of the signal space

    R: int
        Number of samples for resampling procedure

    n_jobs: int, None
        Number of jobs for parallel processing using
        sklearn.externals.joblib.Parallel. If None, will not use parallel
        processing.

    """

    # resample for U and V
    U_norm_samples = norms_sample_project(
        X=X.T, basis=U[:, 0:rank], R=R, n_jobs=n_jobs
    )

    V_norm_samples = norms_sample_project(
        X=X, basis=V[:, 0:rank], R=R, n_jobs=n_jobs
    )

    sigma_min = D[rank - 1]  # TODO: double check -1
    wedin_bound_samples = [
        min(max(U_norm_samples[r], V_norm_samples[r]) / sigma_min, 1)
        for r in range(R)
    ]

    return wedin_bound_samples


def norms_sample_project(X, basis, R=1000, n_jobs=None):
    """
    Samples vectors from space orthogonal to signal space as follows
    - sample random vector from isotropic distribution
    - project onto orthogonal complement of signal space and normalize

    Parameters
    ---------
    X:
        Rhe observed data

    B:
        Rhe basis for the signal col/rows space (e.g. the left/right singular vectors)

    rank: int
        Number of columns to resample

    R: int
        Number of samples


    n_jobs: int, None
        Number of jobs for parallel processing using
        sklearn.externals.joblib.Parallel. If None, will not use parallel
        processing.

    Output
    ------
    an array of the resampled norms
    """

    if n_jobs is not None:
        samples = Parallel(n_jobs=n_jobs)(
            delayed(_get_sample)(X, basis) for i in range(R)
        )
    else:
        samples = [_get_sample(X, basis) for r in range(R)]

    return np.array(samples)


def _get_sample(X, basis):
    dim, rank = basis.shape

    # sample from isotropic distribution
    vecs = np.random.normal(size=(dim, rank))

    # project onto space orthogonal to cols of B
    # vecs = (np.eye(dim) - np.dot(basis, basis.T)).dot(vecs)
    vecs = vecs - np.dot(basis, np.dot(basis.T, vecs))

    # orthonormalize
    vecs, _ = np.linalg.qr(vecs)

    # compute  operator L2 norm
    return np.linalg.norm(X.dot(vecs), ord=2)
