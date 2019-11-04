"""
kcca.py
====================================
Python module for regularized kernel canonical correlation analysis.
Code adopted from UC Berkeley, Gallant lab
(https://github.com/gallantlab/pyrcca)
"""

import numpy as np
from scipy.linalg import eigh


class KCCA(object):
    """
    Kernel CCA class initialization and methods

    Parameters
    ----------
    reg
        Regularization parameter. Default is 0.1. (Float)
    numCC
        Number of canonical dimensions to keep. Default is 10. (Integer)
    kernelcca
        Kernel or non-kernel CCA. Default is True. (Boolean)
    ktype
        Type of kernel if kernelcca is True. (String)
        Value can be 'linear' (default), 'gaussian' or 'polynomial'.
    verbose
        Provides detailed explanation of results. Default is True. (Boolean)
    cutoff
        Optional regularization parameter to perform spectral
        cutoff when computing the canonical weight pseudoinverse
        during held-out data prediction
        Default is 1x10^-15 (Float)
    gausigma
        Parameter if the kernel is a Gaussian kernel. (Float)
    degree
        Parameter if the kernel is a Polynomial kernel. (Integer)

    Returns
    -------
    ws_
        Canonical weights (List)
    comps_
        Canonical components (List)
    cancorrs_
        Correlations of the canonical components on the training dataset (List)
    corrs_
        Correlations on the validation dataset (List)
    preds_
        Predictions on the validation dataset (List)
    ev_
        Explained variance for each canonical dimension (List)
    """

    def __init__(
        self,
        reg=None,
        numCC=None,
        kernelcca=True,
        ktype=None,
        verbose=False,
        cutoff=1e-15,
        gausigma=1.0,
        degree=2,
    ):
        self.reg = reg
        self.numCC = numCC
        self.kernelcca = kernelcca
        self.ktype = ktype
        self.cutoff = cutoff
        self.gausigma = gausigma
        self.degree = degree
        if self.kernelcca and self.ktype is None:
            self.ktype = "linear"
        self.verbose = verbose

    def train(self, data):
        """
        Trains CCA with given parameters

        Parameters
        ----------
        reg
            Regularization parameter. Default is 0.1. (Float)
        numCC
            Number of canonical dimensions to keep. Default is 10. (Integer)
        kernelcca
            Kernel or non-kernel CCA. Default is True. (Boolean)
        ktype
            Type of kernel if kernelcca is True. (String)
            Value can be 'linear' (default), 'gaussian' or 'polynomial'.
        verbose
            Provides detailed explanation of results.
            Default is True. (Boolean)
        cutoff
            Optional regularization parameter to perform spectral cutoff when
            computing the canonical weight pseudoinverse during held-out
            data prediction. Default is 1x10^-15 (Float)
        gausigma
            Parameter if the kernel is a Gaussian kernel. (Float)
        degree
            Parameter if the kernel is a Polynomial kernel. (Integer)

        Returns
        -------
        ws_
            Canonical weights (List)
        comps_
            Canonical components (List)
        cancorrs_
            Correlations of the canonical components on
            the training dataset (List)
        """
        if self.verbose:
            print(
                "Training CCA, kernel = %s, regularization = %0.4f, "
                "%d components" % (self.ktype, self.reg, self.numCC)
            )

        comps_ = kcca(
            data,
            self.reg,
            self.numCC,
            kernelcca=self.kernelcca,
            ktype=self.ktype,
            gausigma=self.gausigma,
            degree=self.degree,
        )
        self.cancorrs_, self.ws_, self.comps_ = recon(
            data, comps_, kernelcca=self.kernelcca
        )

        # self.ev_ = compute_ev(data)
        if len(data) == 2:
            self.cancorrs_ = self.cancorrs_[np.nonzero(self.cancorrs_)]
        return self

    def validate(self, vdata):
        """
        Tests how well the CCA mapping generalizes to the test data
        For each dimension in the test data, correlations between
        predicted and actual data are computed.

        Parameters
        ----------
        vdata
            Standardized data (z-score) (Float)

        Returns
        -------
        corrs_
            Correlations on the validation dataset (List)
        """
        vdata = [np.nan_to_num(_zscore(d)) for d in vdata]
        if not hasattr(self, "ws_"):
            raise NameError("Algorithm has not been trained.")
        self.preds_, self.corrs_ = predict(vdata, self.ws_, self.cutoff)
        return self.corrs_

    def compute_ev(self, vdata):
        """
        Computes the explained variance for each canonical dimension

        Parameters
        ----------
        vdata
            Standardized data (z-score) (Float)

        Returns
        -------
        ev_
            Explained variance for each canonical dimension (List)
        """
        nD = len(vdata)
        nC = self.ws_[0].shape[1]
        nF = [d.shape[1] for d in vdata]
        self.ev_ = [np.zeros((nC, f)) for f in nF]
        for cc in range(nC):
            ccs = cc + 1
            if self.verbose:
                print("Computing explained variance for component #%d" % ccs)
            preds_, corrs_ = predict(
                vdata, [w[:, ccs - 1: ccs] for w in self.ws_], self.cutoff
            )
            resids = [abs(d[0] - d[1]) for d in zip(vdata, preds_)]
            for s in range(nD):
                ev_ = abs(vdata[s].var(0) - resids[s].var(0)) / vdata[s].var(0)
                ev_[np.isnan(ev_)] = 0.0
                self.ev_[s][cc] = ev_
        return self.ev_


def predict(vdata, ws_, cutoff=1e-15):
    """
    Get predictions for each dataset based on the other datasets
    and weights. Find correlations with actual dataset.

    Parameters
    ----------
    vdata
        Standardized data (z-score) (Float)
    ws_
        Canonical weights (List)

    Returns
    -------
    corrs_
        Correlations on the validation dataset (List)
    preds_
        Predictions on the validation dataset (List)
    """
    iws = [np.linalg.pinv(w.T, rcond=cutoff) for w in ws_]
    ccomp = _listdot([d.T for d in vdata], ws_)
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
    return preds_, corrs_


def kcca(
    data, reg=0.0, numCC=None, kernelcca=True,
    ktype="linear", gausigma=1.0, degree=2
):

    """
    Sets up and solves the kernel CCA eigenproblem

    Parameters
    ----------
    data
        Data that kCCA is being run on (Array)
    reg
        Regularization parameter. Default is 0.1. (Float)
    numCC
        Number of canonical dimensions to keep. Default is 10. (Integer)
    kernelcca
        Kernel or non-kernel CCA. Default is True. (Boolean)
    ktype
        Type of kernel if kernelcca is True. (String)
        Value can be 'linear' (default), 'gaussian' or 'polynomial'.
    gausigma
        Parameter if the kernel is a Gaussian kernel. (Float)
    degree
        Parameter if the kernel is a Polynomial kernel. (Integer)

    Returns
    -------
    comp
        Component to determine the canonical weights (Array)

    """
    if kernelcca:
        kernel = [
            _make_kernel(d, ktype=ktype, gausigma=gausigma,
                         degree=degree) for d in data
        ]
    else:
        kernel = [d.T for d in data]

    nDs = len(kernel)
    nFs = [k.shape[0] for k in kernel]
    numCC = min([k.shape[1] for k in kernel]) if numCC is None else numCC

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
    r, Vs = eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for i in range(nDs):
        comp.append(Vs[sum(nFs[:i]): sum(nFs[: i + 1]), :numCC])
    return comp


def recon(data, comp, corronly=False, kernelcca=True):
    """
    Calculates canonical weights, correlations and components

    Parameters
    ----------
    data
        Data of interest (Array)
    comp
        Component to determine the canonical weights (Array)

    Returns
    -------
    corrs_
        Pairwise row correlations for all items in array (List of matrices)
    ws_
        Canonical weights (List)
    ccomp
        Canonical components (List)
    """

    if kernelcca:
        ws_ = _listdot(data, comp)
    else:
        ws_ = comp
    ccomp = _listdot([d.T for d in data], ws_)
    corrs_ = _listcorr(ccomp)
    if corronly:
        return corrs_
    else:
        return corrs_, ws_, ccomp


def _zscore(d):
    """
    Calculates z-score of data

    Parameters
    ----------
    d
        Data of interest (Array)

    Returns
    -------
    z
        Z-score (Array)
    """
    z = (d - d.mean(0)) / d.std(0)
    return z


def _demean(d):
    """
    Calculates difference from mean of the data

    Parameters
    ----------
    d
        Data of interest (Array)

    Returns
    -------
    diff
        Difference from the mean (Array)
    """
    diff = d - d.mean(0)
    return diff


def _listdot(d1, d2):
    """
    Calculates the dot product between two arrays

    Parameters
    ----------
    d1
        Data of interest (Array)
    d1
        Data of interest (Array)

    Returns
    -------
    ld
        Dot product (List)
    """
    ld = [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]
    return ld


def _listcorr(a):
    """
    Returns pairwise row correlations for all items
    in array as a list of matrices

    Parameters
    ----------
    a
        Matrix

    Returns
    -------
    corrs_
        Pairwise row correlations for all items in array (List of matrices)
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
    a
        Matrix row 1
    b
        Matrix row 2

    Returns
    -------
    cs
        Correlations between corresponding rows (Array)

    """
    cs = np.zeros((a.shape[0]))
    for idx in range(a.shape[0]):
        cs[idx] = np.corrcoef(a[idx], b[idx])[0, 1]
    return cs


def _make_kernel(d, normalize=True, ktype="linear", gausigma=1.0, degree=2):
    """
    Makes a kernel for data d
      If ktype is 'linear', the kernel is a linear inner product
      If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = gausigma
      If ktype is 'poly', the kernel is a polynomial kernel with degree=degree

    Parameters
    ----------
    d
        Data (Array)
    ktype
        Type of kernel if kernelcca is True. (String)
        Value can be 'linear' (default), 'gaussian' or 'polynomial'.
    gausigma
        Parameter if the kernel is a Gaussian kernel. (Float)
    degree
        Parameter if the kernel is a Polynomial kernel. (Integer)

    Returns
    -------
    kernel
        Kernel that data is projected to (Array)
    """
    d = np.nan_to_num(d)
    cd = _demean(d)
    if ktype == "linear":
        kernel = np.dot(cd, cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform

        pairwise_dists = squareform(pdist(d, "euclidean"))
        kernel = np.exp(-pairwise_dists ** 2 / 2 * gausigma ** 2)
    elif ktype == "poly":
        kernel = np.dot(cd, cd.T) ** degree
    kernel = (kernel + kernel.T) / 2.0
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel
