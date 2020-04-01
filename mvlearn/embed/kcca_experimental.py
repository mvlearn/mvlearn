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
from sklearn.metrics.pairwise import euclidean_distances

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
        ktype2='linear',
        sigma=1.0,
        degree=2.0,
        constant=1.0,
        test=False
    ):
        self.reg = reg
        self.n_components = n_components
        self.ktype = ktype
        self.sigma = sigma
        self.degree = degree
        self.constant = constant
        if self.ktype is None:
            self.ktype = "linear"
        self.ktype2 = ktype2
        self.test = test

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

        self.Kx = _make_kernel(x, x, self.ktype, self.constant,
                               self.degree, self.sigma)
        self.Ky = _make_kernel(y, y, self.ktype, self.constant,
                               self.degree, self.sigma)

        Id = np.eye(N)
        Z = np.zeros((N, N))

        # Solving eigenvalue problem
        R = 0.5*np.r_[np.c_[self.Kx, self.Ky], np.c_[self.Kx, self.Ky]]
        D = np.r_[np.c_[self.Kx+self.reg*Id, Z], np.c_[Z, self.Ky+self.reg*Id]]

        # Solving eigenvalue problem
        # R = [ 0   KxKy]
        #     [KyKx   0 ]
        # D = []
        R = np.r_[np.c_[Z, self.Ky @ self.Kx], 
                np.c_[self.Kx @ self.Ky, Z]]
        D = np.r_[np.c_[self.Kx @ self.Kx + self.reg * Id, Z],
                np.c_[Z, self.Ky @ self.Ky + self.reg * Id]]

        # # R\alpha = \lambda D \beta
        betas, alphas= linalg.eigh(0.5*(R+R.T), 0.5*(D+D.T))  # eigenvalues, right eigenvectors
        #betas, alphas= linalg.eig(R, D) 
        
        # # Top eigenvalues
        ind = np.argsort(betas)[::-1][:self.n_components]

        # # Extract relevant coordinates and normalize to unit length
        weight1 = alphas[:N, ind]
        weight2 = alphas[N:, ind]

        weight1 /= np.linalg.norm(weight1, axis=0)
        weight2 /= np.linalg.norm(weight2, axis=0)

        self.weights_ = [weight1, weight2]

        if self.test:
            self.Xs = Xs

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

        if not hasattr(self, "Xs"):
            raise NameError("Testing mode was disabled during training")

        Xs = check_Xs(Xs, multiview=True)

        Kx = _make_kernel(_center_norm(Xs[0]), _center_norm(self.Xs[0]), self.ktype, self.constant,
                        self.degree, self.sigma)
        Ky = _make_kernel(_center_norm(Xs[1]), _center_norm(self.Xs[1]), self.ktype, self.constant,
                        self.degree, self.sigma)

        comp1 = Kx @ self.weights_[0]
        comp2 = Ky @ self.weights_[1]

        return([comp1, comp2])

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
    x = x - x.mean(0)
    #x = x - numpy.matlib.repmat(np.mean(x, axis=0), N, 1)
    return x#x@np.sqrt(np.diag(np.divide(1, np.diag(np.transpose(x)@x))))

def _make_kernel(X, Y, ktype, constant=10, degree=2.0, sigma=1.0):
    Nl = len(X)
    Nr = len(Y)
    N0l = np.eye(Nl)#-1/Nl*np.ones(Nl)
    N0r = np.eye(Nr)#-1/Nr*np.ones(Nr)
    # Linear kernel
    if ktype == "linear":
        return N0l@(X@Y.T + constant)@N0r

    # Polynomial kernel
    elif ktype == "poly":
        return N0l@(X@Y.T + constant)**degree@N0r

    # Gaussian kernel
    elif ktype == "gaussian":
        distmat = euclidean_distances(X, Y, squared=True)

        return N0l@np.exp(-distmat/(2*sigma**2))@N0r