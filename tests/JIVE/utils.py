import numpy as np


def svd_checker(U, D, V, n, d, rank):
    checks = {}

    # scores shape
    checks['scores_shape'] = U.shape == (n, rank)

    # scores have orthonormal columns
    checks['scores_ortho'] = np.allclose(np.dot(U.T, U), np.eye(rank))

    # singular values shape
    checks['svals_shape'] = D.shape == (rank, )

    # singular values are in non-increasing order
    svals_nonincreasing = True
    for i in range(len(D) - 1):
        if D[i] < D[i+1]:
            svals_nonincreasing = False
    checks['svals_nonincreasing'] = svals_nonincreasing

    # loadings shape
    checks['loading_shape'] = V.shape == (d, rank)

    # loadings have orthonormal columns
    checks['loadings_ortho'] = np.allclose(np.dot(V.T, V), np.eye(rank))

    return checks
