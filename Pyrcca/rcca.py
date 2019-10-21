"""Python module for regularized kernel canonical correlation analysis"""

import h5py
import numpy as np
from scipy.linalg import eigh

class _CCABase(object):
    def __init__(self, numCV=None, reg=None, regs=None, numCC=None,
                 numCCs=None, kernelcca=True, ktype=None, verbose=False,
                 select=0.2, cutoff=1e-15, gausigma=1.0, degree=2):
        self.numCV = numCV
        self.reg = reg
        self.regs = regs
        self.numCC = numCC
        self.numCCs = numCCs
        self.kernelcca = kernelcca
        self.ktype = ktype
        self.cutoff = cutoff
        self.select = select
        self.gausigma = gausigma
        self.degree = degree
        if self.kernelcca and self.ktype == None:
            self.ktype = 'linear'
        self.verbose = verbose

    def train(self, data):
        if self.verbose:
            print('Training CCA, kernel = %s, regularization = %0.4f, '
                  '%d components' % (self.ktype, self.reg, self.numCC))

        comps_ = kcca(data, self.reg, self.numCC, kernelcca=self.kernelcca,
                     ktype=self.ktype, gausigma=self.gausigma,
                     degree=self.degree)
        self.cancorrs_, self.ws_, self.comps_ = recon(data, comps_,
                                                   kernelcca=self.kernelcca)
        if len(data) == 2:
            self.cancorrs_ = self.cancorrs_[np.nonzero(self.cancorrs_)]
        return self

    def validate(self, vdata):
        vdata = [np.nan_to_num(_zscore(d)) for d in vdata]
        if not hasattr(self, 'ws_'):
            raise NameError('Algorithm has not been trained.')
        self.preds_, self.corrs_ = predict(vdata, self.ws_, self.cutoff)
        return self.corrs_

    def compute_ev(self, vdata):
        nD = len(vdata)
        nC = self.ws_[0].shape[1]
        nF = [d.shape[1] for d in vdata]
        self.ev_ = [np.zeros((nC, f)) for f in nF]
        for cc in range(nC):
            ccs = cc+1
            if self.verbose:
                print('Computing explained variance for component #%d' % ccs)
            preds_, corrs_ = predict(vdata, [w[:, ccs-1:ccs] for w in self.ws_],
                                   self.cutoff)
            resids = [abs(d[0]-d[1]) for d in zip(vdata, preds_)]
            for s in range(nD):
                ev_ = abs(vdata[s].var(0) - resids[s].var(0))/vdata[s].var(0)
                ev_[np.isnan(ev_)] = 0.
                self.ev_[s][cc] = ev_
        return self.ev_

    def save(self, filename):
        h5 = h5py.File(filename, 'a')
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, list):
                    for di in range(len(value)):
                        grpname = 'dataset%d' % di
                        dgrp = h5.require_group(grpname)
                        try:
                            dgrp.create_dataset(key, data=value[di])
                        except RuntimeError:
                            del h5[grpname][key]
                            dgrp.create_dataset(key, data=value[di])
                else:
                    h5.attrs[key] = value
        h5.close()

    def load(self, filename):
        h5 = h5py.File(filename, 'a')
        for key, value in h5.attrs.items():
            setattr(self, key, value)
        for di in range(len(h5.keys())):
            ds = 'dataset%d' % di
            for key, value in h5[ds].items():
                if di == 0:
                    setattr(self, key, [])
                self.__getattribute__(key).append(value.value)


class CCA(_CCABase):
    """Attributes:
        reg (float): regularization parameter. Default is 0.1.
        numCC (int): number of canonical dimensions to keep. Default is 10.
        kernelcca (bool): kernel or non-kernel CCA. Default is True.
        ktype (string): type of kernel used if kernelcca is True.
                        Value can be 'linear' (default) or 'gaussian'.
        verbose (bool): default is True.
    Returns:
        ws_ (list): canonical weights
        comps_ (list): canonical components
        cancorrs_ (list): correlations of the canonical components
                         on the training dataset
        corrs_ (list): correlations on the validation dataset
        preds_ (list): predictions on the validation dataset
        ev_ (list): explained variance for each canonical dimension
    """
    def __init__(self, reg=0., numCC=10, kernelcca=True, ktype=None,
                 verbose=True, cutoff=1e-15):
        super(CCA, self).__init__(reg=reg, numCC=numCC, kernelcca=kernelcca,
                                  ktype=ktype, verbose=verbose, cutoff=cutoff)

    def train(self, data):
        return super(CCA, self).train(data)


def predict(vdata, ws_, cutoff=1e-15):
    """Get predictions for each dataset based on the other datasets
    and weights. Find correlations with actual dataset."""
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


def kcca(data, reg=0., numCC=None, kernelcca=True, ktype='linear',
         gausigma=1.0, degree=2):
    """Set up and solve the kernel CCA eigenproblem
    """
    if kernelcca:
        kernel = [_make_kernel(d, ktype=ktype, gausigma=gausigma,
                               degree=degree) for d in data]
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
        RH[sum(nFs[:i]) : sum(nFs[:i+1]),
           sum(nFs[:i]) : sum(nFs[:i+1])] = (crosscovs[i * (nDs + 1)]
                                             + reg * np.eye(nFs[i]))

        for j in range(nDs):
            if i != j:
                LH[sum(nFs[:j]) : sum(nFs[:j+1]),
                   sum(nFs[:i]) : sum(nFs[:i+1])] = crosscovs[nDs * j + i]

    LH = (LH + LH.T) / 2.
    RH = (RH + RH.T) / 2.

    maxCC = LH.shape[0]
    r, Vs = eigh(LH, RH, eigvals=(maxCC - numCC, maxCC - 1))
    r[np.isnan(r)] = 0
    rindex = np.argsort(r)[::-1]
    comp = []
    Vs = Vs[:, rindex]
    for i in range(nDs):
        comp.append(Vs[sum(nFs[:i]):sum(nFs[:i + 1]), :numCC])
    return comp


def recon(data, comp, corronly=False, kernelcca=True):
    # Get canonical variates and CCs
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

def _zscore(d): return (d - d.mean(0)) / d.std(0)


def _demean(d): return d - d.mean(0)


def _listdot(d1, d2): return [np.dot(x[0].T, x[1]) for x in zip(d1, d2)]


def _listcorr(a):
    """Returns pairwise row correlations for all items in array as a list of matrices
    """
    corrs_ = np.zeros((a[0].shape[1], len(a), len(a)))
    for i in range(len(a)):
        for j in range(len(a)):
            if j > i:
                corrs_[:, i, j] = [np.nan_to_num(np.corrcoef(ai, aj)[0, 1])
                                  for (ai, aj) in zip(a[i].T, a[j].T)]
    return corrs_


def _rowcorr(a, b):
    """Correlations between corresponding matrix rows"""
    cs = np.zeros((a.shape[0]))
    for idx in range(a.shape[0]):
        cs[idx] = np.corrcoef(a[idx], b[idx])[0, 1]
    return cs


def _make_kernel(d, normalize=True, ktype='linear', gausigma=1.0, degree=2):
    """Makes a kernel for data d
      If ktype is 'linear', the kernel is a linear inner product
      If ktype is 'gaussian', the kernel is a Gaussian kernel, sigma = gausigma
      If ktype is 'poly', the kernel is a polynomial kernel with degree=degree
    """
    d = np.nan_to_num(d)
    cd = _demean(d)
    if ktype == 'linear':
        kernel = np.dot(cd, cd.T)
    elif ktype == 'gaussian':
        from scipy.spatial.distance import pdist, squareform
        pairwise_dists = squareform(pdist(d, 'euclidean'))
        kernel = np.exp(-pairwise_dists ** 2 / 2 * gausigma ** 2)
    elif ktype == 'poly':
        kernel = np.dot(cd, cd.T) ** degree
    kernel = (kernel + kernel.T) / 2.
    if normalize:
        kernel = kernel / np.linalg.eigvalsh(kernel).max()
    return kernel
