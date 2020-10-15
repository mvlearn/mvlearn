import numpy as np
from sklearn.utils import check_random_state

from mvlearn.mcca.linalg_utils import rand_orthog


def sample_joint_factor_model(
    n_samples=200,
    n_features=[10, 20, 30],
    joint_rank=3,
    noise_std=1,
    m=1.5,
    random_state=None,
):
    """
    Samples from a low rank, joint factor model where there is one set of shared scores.

    For b = 1, .., B
        X_b = U @ diag(svals) @ W_b^T + noise_std * E_b

    where U and each W_b are orthonormal matrices. The singular values are linearly increasing following (Choi et al. 2017) section 2.2.3.

    Parameters
    -----------
    n_samples: int
        Number of samples.

    n_features: list of ints
        Number of features in each view.

    joint_rank: int
        Rank of the joint signal.

    noise_std: float
        Scale of noise distribution.

    m: float
        Signal strength.

    Output
    ------
    Xs, U_true, view_loadings_true

    Xs: list of array-like
        The data matrices.

    U_true: (n_samples, joint_rank)
        The true orthonormal joint scores matrix.

    view_loadings_true: list of array-like
        The true view loadings matrices.
    """
    rng = check_random_state(random_state)
    n_blocks = len(n_features)

    view_loadings = [rand_orthog(d, joint_rank, random_state=rng) for d in n_features]

    svals = np.arange(1, 1 + joint_rank).astype(float)
    svals *= m * noise_std * (n_samples * max(n_features)) ** (1 / 4)
    U = rng.normal(size=(n_samples, joint_rank))
    U = np.linalg.qr(U)[0]

    Es = [noise_std * rng.normal(size=(n_samples, d)) for d in n_features]
    Xs = [(U * svals) @ view_loadings[b].T + Es[b] for b in range(n_blocks)]

    return Xs, U, view_loadings
