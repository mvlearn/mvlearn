class PermICA(BaseICA):
    r"""
    Performs one ICA per view (ex: subject) and align sources
    using the hungarian algorithm.

    Parameters
    ----------
    n_components : int, optional
        Number of components to extract. If None, no dimension reduction is
        performed and all views must have the same number of features.
    max_iter : int, default=1000
        Maximum number of iterations to perform
    random_state : int, RandomState instance or None, default=None
        Used to perform a random initialization. If int, random_state is
        the seed used by the random number generator; If RandomState
        instance, random_state is the random number generator; If
        None, the random number generator is the RandomState instance
        used by np.random.
    tol : float, default=1e-3
        A positive scalar giving the tolerance at which
        the un-mixing matrices are considered to have converged.
    n_jobs : int (positive), default=None
        The number of jobs to run in parallel. `None` means 1 job, `-1`
        means using all processors.

    Attributes
    ----------
    components_ : np array of shape (n_groups, n_features, n_components)
        The projection matrices that project group data in reduced space.
        Has value None if n_components is None
    unmixings_ : np array of shape (n_groups, n_components, n_components)
        Estimated un-mixing matrices
    source_ : np array of shape (n_samples, n_components)
        Estimated source matrix

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
        random_state=None,
        tol=1e-7,
        n_jobs=None
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, Xs, y=None):
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
        P, Xs = _reduce_data(
            Xs, self.n_components, self.n_jobs
        )
        Xs = np.asarray([X.T for X in Xs])
        n_pb, p, n = Xs.shape
        W = np.zeros((n_pb, p, p))
        S = np.zeros((n_pb, p, n))
        parallelized = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_view_fit)(X) for X in Xs
        )
        S, W = zip(*parallelized)
        W = np.asarray(W)
        S = np.asarray(S)
        orders, signs, S = _find_ordering(S)
        for i, (order, sign) in enumerate(zip(orders, signs)):
            W[i] = sign[:, None] * W[i][order, :]

        self.components_ = P
        self.unmixings_ = np.swapaxes(W, 1, 2)
        self.source_ = S.T

        return self

    def _single_view_fit(self, X):
        Ki, Wi, Si = picard(
            X,
            ortho=False,
            extended=False,
            centering=False,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        scale = np.linalg.norm(Si, axis=1)
        Si = Si / scale[:, None]
        Wi = np.dot(Wi, Ki) / scale[:, None]

        return Si, Wi
