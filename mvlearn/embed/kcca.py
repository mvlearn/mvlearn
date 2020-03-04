"""
kcca.py
====================================
Python module for kernel canonical correlation analysis (kCCA)

Adopted from Steven Van Vaerenbergh's MATLAB Package 'KMBOX'
https://github.com/steven2358/kmbox
"""

from .base import BaseEmbed
from ..utils.utils import check_Xs

import numpy as np
import numpy.matlib
from scipy import linalg


class KCCA(BaseEmbed):
    """
    CCA aims to find useful projections of the
    high- dimensional variable sets onto the compact
    linear representations, the canonical components (components_).

    Each resulting canonical variate is computed from
    the weighted sum of every original variable indicated
    by the canonical weights (weights_).

    The canonical correlation (cancorrs_) quantifies the linear
    correspondence between the two views of data
    based on Pearson’s correlation between their
    canonical components.

    Canonical correlation can be seen as a metric of
    successful joint information reduction between two views
    and, therefore, routinely serves as a performance measure for CCA.

    The kernel generalization of CCA, kernel CCA, is used when
    there are nonlinear relations between two views.
    
    More information to come.

    Parameters
    ----------
    reg : float, default = 0.1
          Regularization parameter
    n_components : int, default = 10
                   Number of canonical dimensions to keep
    ktype : string, default = 'linear'
            Type of kernel
        - value can be 'linear', 'gaussian' or 'polynomial'
    cutoff : float, default = 1x10^-15
             Optional regularization parameter
             to perform spectral cutoff when computing the canonical
             weight pseudoinverse during held-out data prediction
    sigma : float, default = 1.0
            Parameter if Gaussian kernel
    degree : integer, default = 2
             Parameter if Polynomial kernel

    """

    def __init__(
        self,
        reg=0.00001,
        n_components=10,
        ktype='linear',
        sigma=1.0,
        degree=2.0,
        constant=1.0
    ):
        self.reg = reg
        self.n_components = n_components
        self.ktype = ktype
        self.sigma = sigma
        self.degree = degree
        self.constant = constant
        if self.ktype is None:
            self.ktype = "linear"

        # Error Handling
        if self.n_components < 0 or not type(self.n_components) == int:
            raise ValueError("n_components must be a positive integer")
        if ((self.ktype != "linear") and (self.ktype != "poly")
                and (self.ktype != "gaussian")):
            raise ValueError("ktype must be 'linear', 'gaussian', or 'poly'.")
        if self.sigma < 0 or not type(self.sigma) == float:
            raise ValueError("sigma must be positive float")
        if not type(self.degree) == float:
            raise ValueError("degree must be float")
        if self.reg < 0 or not type(self.reg) == float:
            raise ValueError("reg must be positive float")
        if not type(self.constant) == float:
            raise ValueError("constant must be positive float")

    def fit(self, Xs):
        """
        Creates kcca mapping by determining
        canonical weghts from Xs.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data for kcca to fit to.
            Each sample will receive its own embedding.

        Returns
        -------
        weights_ : list of array-likes
                   Canonical weights

        """
        Xs = check_Xs(Xs, multiview=True)

        x = Xs[0]
        y = Xs[1]

        N = len(x)
        x = _center_norm(x)
        y = _center_norm(y)

        self.Kx = _make_kernel(x, self.ktype, self.constant,
                               self.degree, self.sigma)
        self.Ky = _make_kernel(y, self.ktype, self.constant,
                               self.degree, self.sigma)

        Id = np.eye(N)
        Z = np.zeros((N, N))
        dim = min(x.shape[1], y.shape[1])

        # Solving eigenvalue problem
        R = 0.5*np.r_[np.c_[self.Kx, self.Ky], np.c_[self.Kx, self.Ky]]
        D = np.r_[np.c_[self.Kx+self.reg*Id, Z],np.c_[Z, self.Ky+self.reg*Id]]

        betas = linalg.eig(R, D)[0]  # eigenvalues
        alphas = linalg.eig(R, D)[1]  # right eigenvectors
        ind = np.argsort(np.sum(np.diag(betas), axis=0), axis=0)

        perm_mat = np.zeros((len(ind), len(ind)))

        for idx, i in enumerate(ind):
            perm_mat[idx, i] = 1

        alphass = np.real(np.dot(alphas, perm_mat)/np.linalg.norm(alphas))

        # weights
        weight1 = alphass[:N, :dim]
        weight2 = alphass[N:, :dim]
        self.weights_ = [weight1, weight2]

        return self

    def transform(self, Xs):
        """
        Uses KCCA weights to transform Xs into canonical components
        and calculates correlations.

        Parameters
        ----------
        vdata: float
               Standardized data (z-score)

        Returns
        -------
        weights_ : list of array-likes
                   Canonical weights
        preds_: list of array-likes
                Prediction components of test dataset
        corrs_: list of array-likes
                Correlations on the test dataset
        """

        if not hasattr(self, "weights_"):
            raise NameError("kCCA has not been trained.")

        weight1 = self.weights_[0]
        weight2 = self.weights_[1]

        comp1 = []
        comp2 = []

        for i in range(weight1.shape[1]):
            comp1.append(self.Kx@weight1[:, i])
            comp2.append(self.Ky@weight2[:, i])

        comp1 = [l*(-10**18) for l in comp1]
        comp2 = [l*10**18 for l in comp2]

        self.components_ = [comp1, comp2]

        return self

    def fit_transform(self, Xs):
        """
        Fits KCCA mapping with given parameters and transforms Xs
        with the KCCA weights to calculate canonical components
        or projects of the views into a shared embedding.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data for kcca to fit to.
            Each sample will receive its own embedding.

        Returns
        -------
        Xs_transformed : array-like 2D with transformed views
        """

        return self.fit(Xs).transform(Xs)

def _center_norm(x):
    N = len(x)
    x = x - numpy.matlib.repmat(np.mean(x, axis=0), N, 1)
    return x@np.sqrt(np.diag(np.divide(1, np.diag(np.transpose(x)@x))))

def _make_kernel(x, ktype, constant=10, degree=2.0, sigma = 1.0):
    N = len(x)
    N0 = np.eye(N)-1/N*np.ones((N, N))

    # Linear kernel
    if ktype == "linear":
        return N0@(x@x.T + constant)@N0

    # Polynomial kernel
    elif ktype == "poly":
        return N0@(x@x.T + constant)**degree@N0

    # Gaussian kernel
    elif ktype == "gaussian":
        norms1 = np.sum(np.square(x), axis=1)[np.newaxis].T
        norms2 = np.sum(np.square(x), axis=1)

        mat1 = numpy.matlib.repmat(norms1, 1, N)
        mat2 = numpy.matlib.repmat(norms2, N, 1)

        distmat = mat1 + mat2 - 2*x@(x.conj().T)
        return N0@np.exp(-distmat/(2*sigma**2))@N0