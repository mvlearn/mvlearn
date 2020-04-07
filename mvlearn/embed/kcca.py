# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .base import BaseEmbed
from ..utils.utils import check_Xs

import numpy as np
from scipy.linalg import eigh


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
        reg=0.1,
        n_components=10,
        ktype='linear',
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

        # Error Handling
        if self.n_components < 0 or not type(self.n_components) == int:
            raise ValueError("n_components must be a positive integer")
        if ((self.ktype != "linear") and (self.ktype != "poly")
                and (self.ktype != "gaussian")):
            raise ValueError("ktype must be 'linear', 'gaussian', or 'poly'.")
        if self.sigma < 0 or not type(self.sigma) == float:
            raise ValueError("sigma must be positive float")
        if self.degree < 0 or not type(self.degree) == int:
            raise ValueError("degree must be positive int")
        if self.reg < 0 or not type(self.reg) == float:
            raise ValueError("reg must be positive float")
        if self.cutoff < 0 or not type(self.cutoff) == float:
            raise ValueError("cutoff must be positive float")

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

        self.components_ = _listdot([d.T for d in Xs], self.weights_)
        self.cancorrs_ = _listcorr(self.components_)
        self.cancorrs_ = self.cancorrs_[np.nonzero(self.cancorrs_)]

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
        weights_ : list of array-likes
                   Canonical weights
        components_ : list of array-likes
                     Canonical components
        cancorrs_ : list of array-likes
                   Correlations of the canonical components on
                   the training dataset
        """
        Xs = check_Xs(Xs, multiview=True)

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
        Uses the kCCA mapping and generalizes to other data
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
    n_components = (min([k.shape[1] for k in kernel])
                    if n_components is None else n_components)

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
    a : list of array-likes

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
      If ktype is 'linear', kernel is a linear inner product
      If ktype is 'gaussian', kernel is a Gaussian kernel, sigma = sigma
      If ktype is 'poly', kernel is a polynomial kernel with degree=degree

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
    cd = d - d.mean(0)
    if ktype == "linear":
        kernel_ = np.dot(cd, cd.T)
    elif ktype == "gaussian":
        from scipy.spatial.distance import pdist, squareform
        # originally just d and no parentheses in denominator
        pairwise_dists = squareform(pdist(cd, "euclidean"))
        kernel_ = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
    elif ktype == "poly":
        kernel_ = np.dot(cd, cd.T) ** degree
    kernel = (kernel_ + kernel_.T) / 2.0
    kernel = kernel / np.linalg.eigvalsh(kernel).max()  # normalize
    return _zscore(kernel)
