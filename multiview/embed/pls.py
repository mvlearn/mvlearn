# Implements the Partial Least Squares view construction method
# by wrapping the PLS regression function from sklearn. View
# construction generates a new projection of the data.

from sklearn.cross_decomposition import PLSRegression


def PLS_embedding(X, Y, n_components=2, return_embedding=True):
    """
    Calculates the Partial Least Squares embedding of the data X given the
    target (covariates) Y using the NIPALS algorithm implemented by sklearn.

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features) Training vectors, where
        n_samples is the number of samples and n_features is the number of
        predictors.
    Y : array-like, shape = (n_samples, n_targets)
        Target vectors, where n_samples is the number of samples and
        n_targets is the number of response variables.
    return_embedding : boolean, True (default)
        whether to return the embedded data or the projection weights

    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)
    if return_embedding:
        return X @ pls.x_weights_
    else:
        return pls.x_weights_
