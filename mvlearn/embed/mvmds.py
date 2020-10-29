# License: MIT

from .base import BaseEmbed
from ..utils.utils import check_Xs
import warnings
import numpy as np
from sklearn.metrics import euclidean_distances


class MVMDS(BaseEmbed):
    r"""
    An implementation of Classical Multiview Multidimensional Scaling for
    jointly reducing the dimensions of multiple views of data [#1MVMDS]_.
    A Euclidean distance matrix is created for each view, double centered,
    and the k largest common eigenvectors between the matrices are found
    based on the stepwise estimation of common principal components. Using
    these common principal components, the views are jointly reduced and
    a single view of k-dimensions is returned.

    MVMDS is often a better alternative to PCA for multi-view data.
    See the ``tutorials`` in the documentation.

    Parameters
    ----------
    n_components : int (positive), default=2
        Represents the number of components that the user would like to
        be returned from the algorithm. This value must be greater than
        0 and less than the number of samples within each view.

    num_iter: int (positive), default=15
        Number of iterations stepwise estimation goes through.

    dissimilarity : {'euclidean', 'precomputed'}, default='euclidean'
        Dissimilarity measure to use:

        'euclidean':
        Pairwise Euclidean distances between points in the dataset.

        'precomputed':
        Xs is treated as pre-computed dissimilarity matrices.

    Attributes
    ----------
    components_: numpy.ndarray, shape(n_samples, n_components)
        Joint transformed MVMDS components of the input views.

    Notes
    -----

    Classical Multiview Multidimensional Scaling can be broken down into two
    steps. The first step involves calculating the Euclidean Distance matrices,
    :math:`Z_i`, for each of the :math:`k` views and double-centering
    these matrices through the following calculations:

    .. math::
        \Sigma_{i}=-\frac{1}{2}J_iZ_iJ_i

    .. math::
        \text{where }J_i=I_i-{\frac {1}{n}}\mathbb{1}\mathbb{1}^T

    The second step involves finding the common principal components of the
    :math:`\Sigma` matrices. These can be thought of as multiview
    generalizations of the principal components found in principal component
    analysis (PCA) given several covariance matrices. The central hypothesis of
    the common principal component model states that given k normal populations
    (views), their :math:`p` x :math:`p` covariance matrices
    :math:`\Sigma_{i}`, for :math:`i = 1,2,...,k` are simultaneously
    diagonalizable as:

    .. math::
        \Sigma_{i} = QD_i^2Q^T

    where :math:`Q` is the common :math:`p` x :math:`p` orthogonal matrix and
    :math:`D_i^2` are positive :math:`p` x :math:`p` diagonal matrices. The
    :math:`Q` matrix contains all the common principal components. The common
    principal component, :math:`q_j`, is found by solving the minimization
    problem:

    .. math::
        \text{Minimize} \sum_{i=1}^{k}n_ilog(q_j^TS_iq_j)
    .. math::
        \text{Subject to } q_j^Tq_j = 1

    where :math:`n_i` represent the degrees of freedom and :math:`S_i`
    represent sample covariance matrices.

    This class does not support ``MVMDS.transform()`` due to the iterative
    nature of the algorithm and the fact that the transformation is done
    during iterative fitting. Use ``MVMDS.fit_transform()`` to do both
    fitting and transforming at once.

    Examples
    --------
    >>> from mvlearn.embed import MVMDS
    >>> from mvlearn.datasets import load_UCImultifeature
    >>> Xs, _ = load_UCImultifeature()
    >>> print(len(Xs)) # number of samples in each view
    6
    >>> print(Xs[0].shape) # number of samples in each view
    (2000, 76)
    >>> mvmds = MVMDS(n_components=5)
    >>> Xs_reduced = mvmds.fit_transform(Xs)
    >>> print(Xs_reduced.shape)
    (2000, 5)

    References
    ----------
    .. [#1MVMDS] Trendafilov, Nickolay T. “Stepwise Estimation of Common
            Principal Components.” Computational Statistics &amp; Data
            Analysis, vol. 54, no. 12, 2010, pp. 3446–3457.,
            doi:10.1016/j.csda.2010.03.010.

    .. [#2MVMDS] Samir Kanaan-Izquierdo, Andrey Ziyatdinov,
        Maria Araceli Burgueño, Alexandre Perera-Lluna, Multiview: a software
        package for multiview pattern recognition methods, Bioinformatics,
        Volume 35, Issue 16, 15 August 2019, Pages 2877–2879

    """
    def __init__(self, n_components=2, num_iter=15, dissimilarity='euclidean'):

        super().__init__()
        self.components_ = None
        self.n_components = n_components
        self.num_iter = num_iter
        self.dissimilarity = dissimilarity

        if (self.num_iter) <= 0:
            raise ValueError('The number of iterations must be greater than 0')

        if (self.n_components) <= 0:
            raise ValueError('The number of components must be greater than 0 '
                             + 'and less than the number of features')

        if self.dissimilarity not in ['euclidean', 'precomputed']:
            raise ValueError('The parameter `dissimilarity` must be one of \
                {`euclidean`, `precomputed`}')

    def _commonpcs(self, Xs):
        """
        Finds Stepwise Estimation of Common Principal Components as described
        by common Trendafilov implementations based on the following paper:

        https://www.sciencedirect.com/science/article/pii/S016794731000112X

        Parameters
        ----------
        Xs: List of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)

        Returns
        -------
        components: numpy.ndarray, shape(n_samples, n_components)
            Joint transformed MVMDS components of the input views.
        """
        n = p = Xs.shape[1]

        views = len(Xs)

        n_num = np.array([n] * views)/np.sum(np.array([n] * views))

        components = np.zeros((p, self.n_components))

        # Initialized by paper
        pi = np.eye(p)

        s = np.zeros((p, p))

        for i in np.arange(views):
            s = s + (n_num[i] * Xs[i])

        _, e2 = np.linalg.eigh(s)

        # Orders the eigenvalues
        q0 = e2[:, ::-1]

        for i in np.arange(self.n_components):

            # Each q is a particular eigenvalue
            q = q0[:, i]
            q = np.array(q).reshape(len(q), 1)
            d = np.zeros((1, views))

            for j in np.arange(views):

                # Represents mu from the paper.
                d[:, j] = np.dot(np.dot(q.T, Xs[j]), q)

            # stepwise iterations
            for j in np.arange(self.num_iter):
                s2 = np.zeros((p, p))

                for yy in np.arange(views):
                    d2 = n_num[yy] * np.sum(np.array([n] * views))

                    # Dividing by .0001 is to prevent divide by 0 error
                    if d[:, yy] == 0:
                        s2 = s2 + (d2 * Xs[yy] / .0001)

                    else:
                        # Refers to d value from previous iteration
                        s2 = s2 + (d2 * Xs[yy] / d[:, yy])

                # eigenvectors dotted with S matrix and pi
                w = np.dot(s2, q)

                w = np.dot(pi, w)

                q = w / np.sqrt(np.dot(w.T, w))

                for yy in np.arange(views):

                    d[:, yy] = np.dot(np.dot(q.T, Xs[yy]), q)

            # creates next component
            components[:, i] = q[:, 0]
            # initializes pi for next iteration
            pi = pi - np.dot(q, q.T)

        return(components)

    def fit(self, Xs, y=None):
        """
        Calculates dimensionally reduced components by inputting the Euclidean
        distances of each view, double centering them, and using the _commonpcs
        function to find common components between views. Works similarly to
        traditional, single-view Multidimensional Scaling.

        Parameters
        ----------
        Xs: list of array-likes or numpy.ndarray
                - Xs length: n_views
                - Xs[i] shape: (n_samples, n_features_i)
        y : ignored
            Included for API compliance.

        """

        if (self.n_components) > len(Xs[0]):
            self.n_components = len(Xs[0])
            warnings.warn('The number of components you have requested is '
                          + 'greater than the number of samples in the '
                          + 'dataset. ' + str(self.n_components)
                          + ' components were computed instead.')

        Xs = check_Xs(Xs, multiview=True)

        mat = np.ones(shape=(len(Xs), len(Xs[0]), len(Xs[0])))

        # Double centering each view as in single-view MDS

        if (self.dissimilarity == 'euclidean'):

            for i in np.arange(len(Xs)):
                view = euclidean_distances(Xs[i])
                view_squared = np.power(np.array(view), 2)

                J = np.eye(len(view)) - (1/len(view))*np.ones(view.shape)
                B = -(1/2) * J @ view_squared @ J
                mat[i] = B

        # If user wants to input special distance matrix

        elif (self.dissimilarity == 'precomputed'):
            for i in np.arange(len(Xs)):
                if (Xs[i].shape[0] != Xs[i].shape[1]):
                    raise ValueError('The input distance matrix must be '
                                     + 'a square matrix')
                else:
                    view = Xs[i]
                    view_squared = np.power(np.array(view), 2)
                    J = np.eye(len(view)) - (1/len(view))*np.ones(view.shape)
                    B = -(1/2) * J @ view_squared @ J
                    mat[i] = B
        else:
            raise ValueError('The parameter `dissimilarity` must be one of \
                {`euclidean`, `precomputed`}')

        self.components_ = self._commonpcs(mat)

        return self

    def fit_transform(self, Xs, y=None):

        """"
        Embeds data matrix(s) using fitted projection matrices

        Parameters
        ----------

        Xs: list of array-likes or numpy.ndarray
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
            The data to embed based on the fit function.
        y : ignored
            Included for API compliance.

        Returns
        -------
        X_transformed: numpy.ndarray, shape(n_samples, n_components)
            Joint transformed MVMDS components of the input views.
        """
        Xs = check_Xs(Xs)
        self.fit(Xs)

        return self.components_
