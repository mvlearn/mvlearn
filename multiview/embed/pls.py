from sklearn.cross_decomposition import PLSRegression


def PLS_embedding(X, Y, n_components=2, return_projections=True):
    """
    Calculates the Partial Least Squares embedding of the data X given the
    target (covariates) Y. The NIPALS algorithm

    Parameters
    ----------
    X : array-like, shape = (n_samples, n_features) Training vectors, where
        n_samples is the number of samples and n_features is the number of
        predictors.
    Y : array-like, shape = (n_samples, n_targets) Target
        vectors, where n_samples is the number of samples and n_targets is the
        number of response variables.

    """
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, Y)
    if return_projections:
        return X @ pls.x_weights_
    else:
        return pls.x_weights_
