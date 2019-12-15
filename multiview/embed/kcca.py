"""
kcca.py
====================================
Python module for regularized kernel canonical correlation analysis.
Code adopted from UC Berkeley, Gallant lab
(https://github.com/gallantlab/pyrcca)
"""

import numpy as np
from scipy.linalg import eigh
#from scipy import stats


class KCCA(object):
    """
    Kernel CCA class initialization and methods

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
    sigma : float, default = 1 
            Parameter if Gaussian kernel
    degree : integer, default = 2
             Parameter if Polynomial kernel

    """

    def __init__(
        self,
        reg=None,
        n_components=None,
        ktype=None,
        cutoff=1e-15,
        sigma=1.0,
        degree=2,
    ):
        self.reg = reg
        self.n_components = n_components
        self.ktype = ktype
        self.cutoff = cutoff
        self.sigma = sigma
        self.degree = degree
        if self.ktype is None:
            self.ktype = "linear"

    def fit(self, Xs):
        """
        Trains KCCA with given parameters. (Insert more thorough description of what kcca does here)

        Parameters
        ----------
        Xs : list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
            The data for kcca to fit to. 
            Each sample will receive its own embedding.

        Returns
        -------
        weights_ : list of array-likes
                  Canonical weights

        """
        Xs = [np.nan_to_num(_zscore(x)) for x in Xs]
        
        components_ = kcca(
            Xs,
            self.reg,
            self.n_components,
            ktype=self.ktype,
            sigma=self.sigma,
            degree=self.degree,
        )
        
        self.weights_ = _listdot(Xs, components_)

        return self.weights_

    def transform(self, Xs):
        """
        Uses kcca weights to apply develop
        canonical components and correlations

        Parameters
        ----------
        vdata: float
               Standardized data (z-score)

        Returns
        -------
        preds_: list of array-likes
                Prediction components of test dataset
        corrs_: list of array-likes
                Correlations on the test dataset
        """
        Xs = [np.nan_to_num(_zscore(d)) for d in Xs]
        
        if not hasattr(self, "weights_"):
            raise NameError("kCCA has not been trained.")
        
        self.components_ = _listdot([d.T for d in Xs], self.weights_)
        self.cancorrs_ = _listcorr(self.components_)
        self.cancorrs_ = self.cancorrs_[np.nonzero(self.cancorrs_)]
        
        return self

    def fit_transform(self, Xs):
        """
        Trains KCCA with given parameters. (Insert more thorough description of what kcca does here)

        Parameters
        ----------
        Xs : list of array-likes
            - Xs shape: (n_views,)
            - Xs[i] shape: (n_samples, n_features_i)
            The data for kcca to fit to. 
            Each sample will receive its own embedding.

        Returns
        -------
        weights_ : list of array-likes
                   Canonical weights
        components_ : list of array-likes
                     Canonical components
        cancorrs_ : list of array-likes
                   Correlations of the canonical components on 
                   the training dataset
        """
        
        Xs = [np.nan_to_num(_zscore(x)) for x in Xs]
        
        components_ = kcca(
            Xs,
            self.reg,
            self.n_components,
            ktype=self.ktype,
            sigma=self.sigma,
            degree=self.degree,
        )
        
        self.weights_ = _listdot(Xs, components_)
        self.components_ = _listdot([d.T for d in Xs], self.weights_)
        self.cancorrs_ = _listcorr(self.components_)

        if len(Xs) == 2:
            self.cancorrs_ = self.cancorrs_[np.nonzero(self.cancorrs_)]
        return self
    
    def validate(self, vdata):
        """
        Uses the kCCA mapping generalizes to other data
        For each dimension in the test data, correlations between
        predicted and actual data are computed.

        Parameters
        ----------
        vdata: float
               Standardized data (z-score)

        Returns
        -------
        preds_: list of array-likes
                Prediction components of test dataset
        vcorrs_: list of array-likes
                Correlations on the test dataset
        """
        
        vdata = [np.nan_to_num(_zscore(d)) for d in vdata]
        
        if not hasattr(self, "weights_"):
            raise NameError("kCCA has not been trained.")
        
        iws = [np.linalg.pinv(w.T, rcond=self.cutoff) for w in self.weights_]
        ccomp = _listdot([d.T for d in vdata], self.weights_)
        ccomp = np.array(ccomp)
        self.vpreds_ = []
        self.vcorrs_ = []
    
        for dnum in range(len(vdata)):
            idx = np.ones((len(vdata),))
            idx[dnum] = False
            proj = ccomp[idx > 0].mean(0)
            pred = np.dot(iws[dnum], proj.T).T
            pred = np.nan_to_num(_zscore(pred))
            self.vpreds_.append(pred)
            cs = np.nan_to_num(_rowcorr(vdata[dnum].T, pred.T))
            self.vcorrs_.append(cs)
        
        return self.vcorrs_
    
    def compute_ev(self, vdata):
        """
        Computes the explained variance for each canonical dimension

        Parameters
        ----------
        vdata: list of array-likes
               Standardized data (z-score)

        Returns
        -------
        ev_: list of array-likes
             Explained variance for each canonical dimension
        """
        nD = len(vdata)
        nC = self.weights_[0].shape[1]
        nF = [d.shape[1] for d in vdata]
        self.ev_ = [np.zeros((nC, f)) for f in nF]
        for cc in range(nC):
            ccs = cc + 1
            weights = [w[:, ccs - 1: ccs] for w in self.weights_]
            
            iws = [np.linalg.pinv(w.T, rcond=self.cutoff) for w in weights]
            ccomp = _listdot([d.T for d in vdata], weights)
            ccomp = np.array(ccomp)
            preds_ = []
            corrs_ = []
        
            for dnum in range(len(vdata)):
                idx = np.ones((len(vdata),))
                idx[dnum] = False
                proj = ccomp[idx > 0].mean(0)
                pred = np.dot(iws[dnum], proj.T).T
                pred = np.nan_to_num(_zscore(pred))
                preds_.append(pred)
                cs = np.nan_to_num(_rowcorr(vdata[dnum].T, pred.T))
                corrs_.append(cs)
            
            resids = [abs(d[0] - d[1]) for d in zip(vdata, preds_)]
            for s in range(nD):
                ev_ = abs(vdata[s].var(0) - resids[s].var(0)) / vdata[s].var(0)
                ev_[np.isnan(ev_)] = 0.0
                self.ev_[s][cc] = ev_
        return self.ev_


def kcca(
    Xs, reg=0.0, n_components=None,
    ktype="linear", sigma=1.0, degree=2
):

    """
    Sets up and solves the kernel CCA eigenproblem

    Parameters
    ----------
    Xs : list of array-likes
        - Xs shape: (n_views,)
        - Xs[i] shape: (n_samples, n_features_i)
        The data for kcca to fit to. 
        Each sample will receive its own embedding.


    Returns
    -------
    comp : list of array-likes
           Component to determine the canonical weights
    """

    kernel = [
        _make_kernel(d, ktype=ktype, sigma=sigma,
                     degree=degree) for d in Xs
    ]

    nDs = len(kernel)
    nFs = [k.shape[0] for k in kernel]
    n_components = min([k.shape[1] for k in kernel]) if n_components is None else n_components

    # Get the auto- and cross-covariance matrices
    crosscovs = [np.dot(ki, kj.T) for ki in kernel for kj in kernel]

    # Allocate left-hand side (LH) and right-hand side (RH):
    LH = np.zeros((sum(nFs), sum(nFs)))
    RH = np.zeros((sum(nFs), sum(nFs)))

    # Fill the left and right sides of the eigenvalue problem
    for i in range(nDs):
        RH[
            sum(nFs[:i]): sum(nFs[: i + 1]), sum(nFs[:i]): sum(nFs[: i + 1])
        ] = crosscovs[i * (nDs + 1)] + reg * np.eye(nFs[i])

        for j in range(nDs):
            if i != j:
                LH[
                    sum(nFs[:j]): sum(nFs[: j + 1]),
                    sum(nFs[:i]): sum(nFs[: i + 1])
                ] = crosscovs[nDs * j + i]

    LH = (LH + LH.T) / 2.0
    RH = (RH + RH.T) / 2.0

    maxCC = LH.shape[0]
    r, Vs = eigh(LH, RH, eigvals=(maxCC - n_components, maxCC - 1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for i in range(nDs):
        comp.append(Vs[sum(nFs[:i]): sum(nFs[: i + 1]), :n_components])
    return comp


def _zscore(d):
    """
    Calculates z-score of data

    Parameters
    ----------
    d : array
        Data of interest

    Returns
    -------
    z : array
        Z-score
    """
    z = (d - d.mean(0)) / d.std(0)
    return z


def _demean(d):
    """
    Calculates difference from mean of the data

    Parameters
    ----------
    d : array
        Data of interest

    Returns
    -------
    diff : array
           Difference from the mean (Array)
    """
    diff = d - d.mean(0)
    return diff


def _listdot(d1, d2):
    """
    Calculates the dot product between two arrays

    Parameters
    ----------
    d1 : array
         Data of interest
    d1 : array
         Data of interest

    Returns
    -------
    ld : list
        Dot product
    """
    ld = [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]
    return ld


def _listcorr(a):
    """
    Returns pairwise row correlations for all items
    in array as a list of matrices

    Parameters
    ----------
    a : Matrix

    Returns
    -------
    corrs_ : list of array-likes
             Pairwise row correlations for all items in array
    """
    corrs_ = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                corrs_[:, i, j] = [
                    np.nan_to_num(np.corrcoef(ai, aj)[0, 1])
                    for (ai, aj) in zip(a[i].T, a[j].T)
                ]
    return corrs_


def _rowcorr(a, b):
    """
    Finds correlations between corresponding matrix rows (a and b)

    Parameters
    ----------
    a : array 
        Matrix row 1
    b : array
        Matrix row 2

    Returns
    -------
    cs: array
        Correlations between corresponding rows

    """
    cs = np.zeros((a.shape[0]))
    for idx in range(a.shape[0]):
        cs[idx] = np.corrcoef(a[idx], b[idx])[0, 1]
    return cs


def _make_kernel(d, normalize=True, ktype="linear", sigma=1.0, degree=2):
    """
    Makes a kernel for data d
      If ktype is 'linear', the kernel is a linear inner product
      If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = sigma
      If ktype is 'poly', the kernel is a polynomial kernel with degree=degree

    Parameters
    ----------
    d : array
        Data
    ktype : string, default = 'linear'
        - Type of kernel
        - Value can be 'linear', 'gaussian' or 'polynomial'.
    sigma : float, default = 1.0
            Parameter if the kernel is a Gaussian kernel.
    degree : int, default = 2
             Parameter if the kernel is a Polynomial kernel. 

    Returns
    -------
    kernel: array
            Kernel that data is projected to
    """
    d = np.nan_to_num(d)
    cd = _demean(d)
    if ktype == "linear":
        kernel_ = np.dot(cd, cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform
        # pdist finds pairwise distances between observations in n-dimensional space
        
        pairwise_dists = squareform(pdist(cd, "euclidean")) #originally just d
        kernel_ = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2)) #originally no parentheses in denom
    elif ktype == "poly":
        kernel_ = np.dot(cd, cd.T) ** degree
    kernel = (kernel_ + kernel_.T) / 2.0
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return _zscore(kernel)
