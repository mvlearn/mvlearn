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
    r"""
    The kernel canonical correlation analysis (KCCA) is a method
    that generalizes the classical linear canonical correlation
    analysis (CCA) to nonlinear setting.  It allows us to depict the
    nonlinear relation of two sets of variables and enables
    applications of classical multivariate data analysis
    originally constrained to linearity relation (CCA).

    Parameters
    ----------
    reg : float, default = 0.00001
          Regularization parameter
    n_components : int, default = 10
                   Number of canonical dimensions to keep
    ktype : string, default = 'linear'
            Type of kernel
        - value can be 'linear', 'gaussian' or 'poly'
    sigma : float, default = 1.0
            Parameter if Gaussian kernel
    degree : float, default = 2.0
             Parameter if Polynomial kernel
    constant : float, default = 1.0
             Parameter if Polynomial kernel

    Notes
    -----
    This class implements kernel canonical correlation analysis
    as described in [#1KCCA]_ and [#2KCCA]_.
    
    Traditional CCA aims to find useful projections of the
    high- dimensional variable sets onto the compact
    linear representations, the canonical components (components_).

    Each resulting canonical variate is computed from
    the weighted sum of every original variable indicated
    by the canonical weights (weights_).

    The canonical correlation quantifies the linear
    correspondence between the two views of data
    based on Pearson’s correlation between their
    canonical components.

    Canonical correlation can be seen as a metric of
    successful joint information reduction between two views
    and, therefore, routinely serves as a performance measure for CCA.

    CCA may not extract useful descriptors of the data because of
    its linearity. kCCA offers an alternative solution by first
    projecting the data onto a higher dimensional feature space.

    .. math::
        \phi: \mathbf{x} = (x_1,...,x_m) \mapsto
        \phi(\mathbf{x}) = (\phi(x_1),...,\phi(x_N)),
        (m < N)

    before performing CCA in the new feature space.

    Kernels are methods of implicitly mapping data into a higher
    dimensional feature space, a method known as the kernel trick.
    A kernel function K, such that for all :math:`\mathbf{x},
    \mathbf{z} \in X`,

    .. math::
        K(\mathbf{x}, \mathbf{z}) = \langle\phi(\mathbf{x})
        \cdot \phi(\mathbf{z})\rangle,

    where :math:`\phi` is a mapping from X to feature space F.

    The directions :math:`\mathbf{w_x}` and :math:`\mathbf{w_y}`
    (of length N) can be rewritten as the projection of the data
    onto the direction :math:`\alpha` and :math:`\alpha`
    (of length m):

    .. math::
        \mathbf{w_x} = X'\alpha
        \mathbf{w_y} = Y'\beta

    Letting :math:`K_x = XX'` and :math:`K_x = XX'` be the kernel
    matrices and adding a regularization term (:math:`\kappa`)
    to prevent overfitting, we are effectively solving for:

    .. math::
        \rho = \underset{\alpha,\beta}{\text{max}}
        \frac{\alpha'K_xK_y\beta}
        {\sqrt{(\alpha'K_x^2\alpha+\kappa\alpha'K_x\alpha)
        \cdot (\beta'K_y^2\beta + \kappa\beta'K_y\beta)}}


    References
    ----------
    .. [#1KCCA] D. R. Hardoon, S. Szedmak and J. Shawe-Taylor,
            "Canonical Correlation Analysis: An Overview with
            Application to Learning Methods", Neural Computation,
            Volume 16 (12), Pages 2639--2664, 2004.
    .. [#2KCCA] Su-Yun Huang, Mei-Hsien Lee and Chuhsing Kate Hsiao,
            "Kernel Canonical Correlation Analysis and its Applications
            to Nonlinear Measures of Association and Test of Independence",
            draft, May 25, 2006
    """

    def __init__(
        self,
        reg=0.1,
        n_components=2,
        ktype='linear',
        sigma=1.0,
        degree=2.0,
        decomp='full',
        method='kettenring-like',
        mrank = 100
    ):
        self.reg = reg
        self.n_components = n_components
        self.ktype = ktype
        self.sigma = sigma
        self.degree = degree
        self.decomp = decomp
        self.method = method
        self.mrank = mrank
        if self.ktype is None:
            self.ktype = "linear"

        # Error Handling
        if self.n_components < 0 or not type(self.n_components) == int:
            raise ValueError("n_components must be a positive integer")
        if ((self.ktype != "linear") and (self.ktype != "poly")
                and (self.ktype != "gaussian")):
            raise ValueError("ktype must be 'linear', 'gaussian', or 'poly'.")
        if self.sigma < 0 or not (type(self.sigma) == float
                                  or type(self.sigma) == int):
            raise ValueError("sigma must be positive int/float")
        if not (type(self.degree) == float or type(self.sigma) == int):
            raise ValueError("degree must be int/float")
        if self.reg < 0 or self.reg > 1 or not type(self.reg) == float:
            raise ValueError("reg must be positive float")

    def fit(self, Xs, y=None):
        r"""
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
        self : returns an instance of self

        """
        Xs = check_Xs(Xs, multiview=True)

        self.X = _center_norm(Xs[0])
        self.Y = _center_norm(Xs[1])

        N = len(self.X)

        if self.decomp == "full":
            Kx = _make_kernel(self.X, self.X, self.ktype,
                                   self.degree, self.sigma)
            Ky = _make_kernel(self.Y, self.Y, self.ktype,
                                   self.degree, self.sigma)

            Id = np.eye(N)
            Z = np.zeros((N, N))
    
            # Method 1: Standard Hardoon
            if self.method == "standard_hardoon":
                R = np.r_[np.c_[Z, Kx@Ky], np.c_[Ky@Kx, Z]]
                D = 0.5*np.r_[np.c_[Kx@(Kx+self.reg*Id), Z],
                              np.c_[Z, Ky@(Ky+self.reg*Id)]]
                R = R/2+R.T/2
                D = D/2+D.T/2
    
            # Method 2: Simplified Hardoon
            elif self.method == "simplified_hardoon":
                R = np.r_[np.c_[Z, Ky], np.c_[Kx, Z]]
                D = np.r_[np.c_[Kx+self.reg*Id, Z], np.c_[Z, Ky+self.reg*Id]]
                R = R/2+R.T/2
                D = D/2+D.T/2
    
            # Method 3: Kettenring-like generalizable formulation
            elif self.method == "kettenring-like":
                R = 0.5*np.r_[np.c_[Kx, Ky], np.c_[Kx, Ky]]
                D = np.r_[np.c_[Kx+self.reg*Id, Z], np.c_[Z, Ky+self.reg*Id]]
        
        elif self.decomp == "icd":            
            G1 = _make_icd_kernel(self.X, self.X, self.ktype, self.degree,
                                  self.sigma, self.mrank)
            G2 = _make_icd_kernel(self.Y, self.Y, self.ktype, self.degree,
                                  self.sigma, self.mrank)
            
            G1 = G1 - numpy.matlib.repmat(np.mean(G1, axis=0), N, 1)
            G2 = G2 - numpy.matlib.repmat(np.mean(G2, axis=0), N, 1)
            
            N1 = len(G1[0])
            N2 = len(G2[0])
            Z11 = np.zeros(N1)
            Z22 = np.zeros(N2)
            Z12 = np.zeros((N1,N2))
            I11 = np.eye(N1)
            I22 = np.eye(N2)

            # Method 1: Standard Hardoon
            if self.method == "standard_hardoon":
                R = np.r_[np.c_[Z11, G1.T@G1@G1.T@G2], 
                          np.c_[G2.T@G2@G2.T@G1, Z22]]
                D = np.r_[np.c_[G1.T@G1@G1.T@G1+self.reg*I11, Z12],
                              np.c_[Z12.T, G2.T@G2@G2.T@G2+self.reg*I22]]
    
            # Method 2: Simplified Hardoon
            elif self.method == "simplified_hardoon":
                R = np.r_[np.c_[Z11, G1.T@G2], np.c_[G2.T@G1, Z22]]
                D = np.r_[np.c_[G1.T@G1+self.reg*I11, Z12],
                          np.c_[Z12.T, G2.T@G2+self.reg*I22]]
    
            # Method 3: Kettenring-like generalizable formulation
            elif self.method == "kettenring-like":
                R = 0.5*np.r_[np.c_[G1.T@G1, G1.T@G2],
                              np.c_[G2.T@G1, G2.T@G2]]
                D = np.r_[np.c_[G1.T@G1+self.reg*I11, Z12],
                          np.c_[Z12.T, G2.T@G2+self.reg*I22]]

        # Solve eigenvalue problem
        betas, alphas = linalg.eig(R, D)

        # Top eigenvalues
        ind = np.argsort(betas)[::-1][:self.n_components]

        # Extract relevant coordinates and normalize to unit length
        weight1 = alphas[:N, ind]
        weight2 = alphas[N:, ind]

        weight1 /= np.linalg.norm(weight1, axis=0)
        weight2 /= np.linalg.norm(weight2, axis=0)

        self.weights_ = np.real([weight1, weight2])

        return self

    def transform(self, Xs):
        r"""
        Uses KCCA weights to transform Xs into canonical components
        and calculates correlations.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: 2
             - Xs[i] shape: (n_samples, n_features_i)
            The data for kcca to fit to.
            Each sample will receive its own embedding.

        weights_ : list of array-likes
                   Canonical weights

        Returns
        -------
        components_ : returns Xs_transformed, a list of numpy.ndarray
             - Xs length: 2
             - Xs[i] shape: (n_samples, n_samples)
        """

        if not hasattr(self, "weights_"):
            raise NameError("kCCA has not been trained.")
        
        Xs = check_Xs(Xs, multiview=True)
        
        #error if number of subjects does not equal fit number of subjects

        Kx_transform = _make_kernel(_center_norm(Xs[0]),
                                    _center_norm(self.X),
                                    self.ktype,
                                    self.degree,
                                    self.sigma)
        Ky_transform = _make_kernel(_center_norm(Xs[1]),
                                    _center_norm(self.Y),
                                    self.ktype,
                                    self.degree,
                                    self.sigma)

        weight1 = self.weights_[0]
        weight2 = self.weights_[1]

        comp1 = []
        comp2 = []

        for i in range(weight1.shape[1]):
            comp1.append(Kx_transform@weight1[:, i])
            comp2.append(Ky_transform@weight2[:, i])

        comp1 = np.transpose(np.asarray(comp1))
        comp2 = np.transpose(np.asarray(comp2))

        self.components_ = [comp1, comp2]

        return self.components_


def _center_norm(x):
    x = x - x.mean(0)
    return x


def _make_kernel(X, Y, ktype, degree=2.0, sigma=1.0):
    Nl = len(X)
    Nr = len(Y)
    N0l = np.eye(Nl) - 1 / Nl * np.ones(Nl)
    N0r = np.eye(Nr) - 1 / Nr * np.ones(Nr)
    
    # Linear kernel
    if ktype == "linear":
        return N0l @ (X @ Y.T) @ N0r

    # Polynomial kernel
    elif ktype == "poly":
        return N0l @ (X @ Y.T) ** degree @ N0r

    # Gaussian kernel
    elif ktype == "gaussian":
        distmat = euclidean_distances(X, Y, squared=True)

        return N0l @ np.exp(-distmat / (2 * sigma ** 2)) @ N0r
    
    # Gaussian diagonal kernel
    elif ktype == "gaussian-diag":
        return np.exp(-np.sum(np.power((X-Y),2), axis=1)/(2*sigma**2))


def _make_icd_kernel(x, ktype, degree=2.0, sigma=1.0, mrank=100):
    N = len(x)
    #precision = 0.000001
    mrank = N
    ktype = "gaussian-diag"
    sigma = 1.0

    perm = np.arange(N)
    d = np.zeros((1,N))
    G = np.zeros((N,mrank))
    subset = np.zeros((1,mrank))

    for i in range(mrank):
        x_new = x[perm[i:N],:]
        if i == 0:
            d[i:N] = _make_kernel(x_new,x_new, ktype)
        else:
            d[i:N] = (_make_kernel(x_new,x_new, ktype) 
                    - np.sum(np.power(G[i:N,:i-1],2),axis=1))

        j = np.argmax(d[i:N])
        m2 = np.max(d[i:N])
        j = j+i-1
        m1 = np.sqrt(m2)
        subset[i] = j

        perm[i], perm[j] = perm[j], perm[i]
        G[i, :i-1], G[j, :i-1] = G[j, :i-1], G[i, :i-1]
        G[i,i] = m1

        G[i+1:N,i] = ((_make_kernel(x[perm[i],:],x[perm[i+1:N],:],ktype,sigma).T 
                         - (G[i+1:N,:i-1]@(G[i,:i-1].T)))/m1)

    ind = np.argsort(perm)
    G = G[ind,:]
