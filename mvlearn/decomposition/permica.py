from .baseica import BaseICA
from multiviewica import permica


class PermICA(BaseICA):
    r"""
    Performs one ICA per view (ex: subject) and align sources
    using the hungarian algorithm.

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If None, n_components is set to
        the minimum number of features in the dataset.

    max_iter : int, default=1000
        Maximum number of iterations to perform

    preproc: 'pca' or a ViewTransformer-like instance,
        default='pca'
        Preprocessing method to use to reduce data.
        If "pca", performs PCA separately on each view to reduce dimension
        of each view.
        Otherwise the dimension reduction is performed using the transform
        method of the ViewTransformer-like object. This instance also needs
        an inverse transform method to recover original data from reduced data.

    multiview_output : bool, optional (default True)
        If True, the `.transform` method returns one dataset per view.
        Otherwise, it returns one dataset, of shape (n_samples, n_components)

    random_state : int, RandomState instance or None, default=None
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.

    tol : float, default=1e-3
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.

    verbose : bool, default=False
        Print information

    Attributes
    ----------
    preproc : ViewTransformer-like instance
        The fitted instance used for preprocessing
    W_list : np array of shape (n_groups, n_components, n_components)
        The unmixing matrices to apply on preprocessed data

    See also
    --------
    groupica
    multiviewica

    References
    ----------
    .. [#1permica] Hugo Richard, Luigi Gresele, Aapo Hyvärinen, Bertrand
        Thirion, Alexandre Gramfort, Pierre Ablin. Modeling Shared Responses
        in Neuroimaging Studies through MultiView ICA. arXiv 2020.

    Examples
    --------
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> from mvlearn.decomposition import PermICA
    >>> Xs, _ = load_UCImultifeature()
    >>> ica = PermICA(n_components=3)
    >>> sources = ica.fit_transform(Xs)
    >>> print(sources.shape)
    (6, 2000, 3)
    """

    def __init__(
        self,
        n_components=None,
        max_iter=1000,
        multiview_output=True,
        random_state=None,
        tol=1e-7,
        verbose=False,
        preproc="pca",
    ):
        super().__init__(
            n_components=n_components,
            preproc=preproc,
            random_state=random_state,
            verbose=verbose,
            multiview_output=multiview_output,
        )
        self.max_iter = max_iter
        self.tol = tol

    def fit_(self, Xs, y=None):
        r"""
        Fits the model to the views Xs.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            Training data to recover a source and unmixing matrices from.
        y : ignored

        Returns
        -------
        self : returns an instance of itself.
        """
        _, W, S = permica(
            Xs,
            max_iter=self.max_iter,
            random_state=self.random_state,
            tol=self.tol,
        )
        return W, S
