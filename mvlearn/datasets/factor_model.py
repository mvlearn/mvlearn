import numpy as np
from sklearn.utils import check_random_state
from ..utils import rand_orthog, param_as_list


def sample_joint_factor_model(
    n_views,
    n_samples,
    n_features,
    joint_rank,
    noise_std=1,
    m=1.5,
    random_state=None,
    return_decomp=False,
):
    """
    Samples from a low rank, joint factor model where there is one set of
    shared scores.

    Parameters
    -----------
    n_views : int
        Number of views to sample

    n_samples : int
        Number of samples in each view

    n_features: int, or list of ints
        Number of features in each view. A list specifies a different number
        of features for each view.

    joint_rank : int
        Rank of the common signal across views.

    noise_std : float
        Scale of noise distribution.

    m : float
        Signal strength.

    Returns
    -------
    Xs : list of array-likes or numpy.ndarray
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        List of samples data matrices

    U: (n_samples, joint_rank)
        The true orthonormal joint scores matrix. Returned if
        `return_decomp` is True.

    view_loadings: list of numpy.ndarray
        The true view loadings matrices. Returned if
        `return_decomp` is True.

    Notes
    -----
    For b = 1, .., B
        X_b = U @ diag(svals) @ W_b^T + noise_std * E_b

    where U and each W_b are orthonormal matrices. The singular values are
    linearly increasing following (Choi et al. 2017) section 2.2.3.
    """
    rng = check_random_state(random_state)
    n_features = param_as_list(n_features, n_views)

    view_loadings = [rand_orthog(d, joint_rank, random_state=rng)
                     for d in n_features]

    svals = np.arange(1, 1 + joint_rank).astype(float)
    svals *= m * noise_std * (n_samples * max(n_features)) ** (1 / 4)
    U = rng.normal(size=(n_samples, joint_rank))
    U = np.linalg.qr(U)[0]

    Es = [noise_std * rng.normal(size=(n_samples, d)) for d in n_features]
    Xs = [(U * svals) @ view_loadings[b].T + Es[b] for b in range(n_views)]

    if return_decomp:
        return Xs, U, view_loadings
    else:
        return Xs
