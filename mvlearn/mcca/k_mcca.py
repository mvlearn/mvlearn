import numpy as np
from itertools import combinations
from warnings import warn
from textwrap import dedent
from sklearn.base import BaseEstimator, TransformerMixin

from mvlearn.utils import check_Xs

from mvlearn.mcca.block_processing import (
    center_kernel_blocks,
    split,
    initial_svds,
    process_block_kernel_args,
    _block_kern_docs,
)
from mvlearn.mcca.mcca import check_regs, mcca_det_output, _mcca_docs
from mvlearn.mcca.linalg_utils import eigh_wrapper, normalize_cols
from mvlearn.mcca.MCCABlock import KMCCABlock


class KMCCA(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_components=1,
        center=True,
        regs=None,
        kernel="linear",
        kernel_params={},
        precomp_svds=None,
        signal_ranks=None,
        sval_thresh=1e-3,
        diag_mode="A",
        # inform=False,
        filter_params=False,
        n_jobs=None,
    ):

        self.n_components = n_components
        self.center = center
        self.regs = regs
        self.kernel = kernel
        self.kernel_params = kernel_params
        self.signal_ranks = signal_ranks
        self.precomp_svds = precomp_svds
        self.sval_thresh = sval_thresh
        self.diag_mode = diag_mode
        # self.inform = inform
        self.filter_params = filter_params
        self.n_jobs = n_jobs

    def fit(self, Xs):
        """
        Fits the regularized kernel MCCA model.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
            The list of data matrices each shaped (n_samples, n_features_b).
        """
        Xs = check_Xs(Xs, multiview=True, return_dimensions=False)
        n_blocks = len(Xs)
        # set up kernels
        kernel, kernel_params = process_block_kernel_args(
            n_blocks=n_blocks, kernel=self.kernel, kernel_params=self.kernel_params
        )

        self.blocks_ = [None for b in range(n_blocks)]
        for b in range(n_blocks):
            self.blocks_[b] = KMCCABlock(
                kernel=kernel[b],
                kernel_params=kernel_params[b],
                filter_params=self.filter_params,
                n_jobs=self.n_jobs,
            )

        Ks = [self.blocks_[b]._get_kernel(Xs[b]) for b in range(n_blocks)]

        # if self.inform:
        #     assert self.diag_mode == 'A'

        #     out = ik_mcca(Ks,
        #                   n_components=self.n_components,
        #                   center=self.center,
        #                   regs=self.regs,
        #                   signal_ranks=self.signal_ranks,
        #                   sval_thresh=self.sval_thresh,
        #                   precomp_svds=precomp_svds,
        #                   method='auto')

        out = k_mcca_evp_svd(
            Ks,
            n_components=self.n_components,
            center=self.center,
            regs=self.regs,
            signal_ranks=self.signal_ranks,
            sval_thresh=self.sval_thresh,
            diag_mode=self.diag_mode,
            precomp_svds=self.precomp_svds,
        )

        self.common_norm_scores_ = out["common_norm_scores"]
        self.cs_col_norms_ = out["cs_col_norms"]
        self.evals_ = out["evals"]

        for b in range(n_blocks):
            dv = out["dual_vars"][b]
            bs = out["block_scores"][b]
            cent = out["centerers"][b]
            self.blocks_[b].set_fit_values(
                dual_vars=dv, block_scores=bs, centerer=cent, X_fit=Xs[b]
            )

        return self

    @property
    def n_blocks_(self):
        if hasattr(self, "blocks_"):
            return len(self.blocks_)

    @property
    def block_scores_(self):
        if hasattr(self, "blocks_"):
            return [blck.block_scores_ for blck in self.blocks_]

    @property
    def n_components_(self):
        if hasattr(self, "common_norm_scores_"):
            return self.common_norm_scores_.shape[1]

    def fit_transform(self, Xs, precomp_svds=None):
        self.fit(Xs, precomp_svds=precomp_svds)
        return self.common_norm_scores_

    def transform(self, Xs):
        common_proj = sum(
            self.blocks_[b].transform(Xs[b]) for b in range(self.n_blocks_)
        )
        return common_proj * (1 / self.cs_col_norms_)


_kmcca_docs = dict(
    Ks=dedent(
        """
    Ks: list of array-like
        List of kernel matrices each shaped (n_samples, n_samples)
    """
    ),
    basic=dedent(
        """
        n_components: int, None
            Number of components to compute.

        center: bool, list
            Whether or not to initially mean center the data. Different options for each data view can be provided by inputting a list of bools.

        regs: None, float, list
            MCCA regularization for each data block, which can be important for kernel methods. A value of 0 or None for all blocks corresponds to SUMCORR-AVGVAR MCCA. A value of 1 corresponds to partial least squares SVD in the case of 2 blocks and a natural generalization of this method for more than two blocks. If a single value (None, float or str) is passed in that value will be used for every block. Different options for each data view can be provided by inputting a list.

    """
    ),
    diag_mode=dedent(
        """
        diag_mode: str
        What to put on the diagonal.
        Must be one of ['A', 'B', 'C']

        'A': TODO: refeence
            (1 - r) * K^2 + r * K

        'B' (Bach and Jordan, 2002)
            (1-r) (K + n/2 kappa * I)^2 where kappa = r / (1 - r)

        'C' TODO: reference other than (Bilenko and Gallant, 2016)
            (1 - r) K^2 + r * I

    """
    ),
    sval_killing=dedent(
        """
    sval_thresh: float
        For each block we throw out singular values of (1/n)K that are too small (i.e. zero or essentially zero). Setting this value to be non-zero is how we deal with the singular block gram matrices.

    signal_ranks: None, int, list
        Largest SVD rank to compute for each block.
    """
    ),
    centerers=dedent(
        """
        centerers: list of sklearn.preprocessing.KernelCenterer
            The mean centering object for each block.
    """
    ),
    kern_basic=_block_kern_docs["basic"],
    kern_other=_block_kern_docs["other"],
    score_out=_mcca_docs["score_out"],
)

KMCCA.__doc__ = dedent(
    """
    Kernel multi-view canonical correlation analysis. Includes options for regularized kernel MCCA and informative kernel MCCA (i.e. where we first compute a low rank kernel PCA).

    Parameters
    -----------
    {basic}

    {kern_basic}

    {diag_mode}

    {sval_killing}

    {kern_other}

    precomp_svds: None, list of tuples
            (optional) Precomputed SVDs of the kernel matrices.

    Attributes
    ----------
    blocks_: list of mvdr.mcca.MCCABlock.KMCCABlock
        Containts the view level data for each data view.

    evals_: array-like, (n_components, )
            The MCCA eigenvalues.

    common_norm_scores_: array-like, (n_samples, n_components)
        Normalized sum of the block scores.

    cs_col_norms_: array-like, (n_components, )
        Column nomrs of the sum of the block scores.
        Useful for projecting new data.

    """.format(
        **_kmcca_docs
    )
)


def k_mcca_evp_svd(
    Ks,
    n_components=None,
    center=True,
    regs=None,
    signal_ranks=None,
    sval_thresh=1e-3,
    diag_mode="A",
    precomp_svds=None,
):

    Ks, n_blocks, n_samples, _ = check_Xs(Ks, multiview=True, return_dimensions=True)
    if sval_thresh is not None:
        # put sval_thresh on the scale of (1/n) K.
        # since we compute SVD of K, put _sval_thresh on scale of svals of K
        _sval_thresh = sval_thresh * n_samples
    else:
        _sval_thresh = None

    # center data blocks
    Ks, centerers = center_kernel_blocks(Ks, center=center)

    #######################
    # Compute initial SVD #
    #######################

    reduced, init_svds, _ = initial_svds(
        Xs=Ks,
        signal_ranks=signal_ranks,
        center=False,
        normalized_scores=True,
        precomp_svds=precomp_svds,
        sval_thresh=_sval_thresh,
    )

    n_features_reduced = [r.shape[1] for r in reduced]
    if n_components is None:
        n_components = sum(n_features_reduced)
    if n_components > sum(n_features_reduced):
        warn("Requested too many components!")
    n_components = min(n_components, sum(n_features_reduced))

    # get singular values of transformed block gram matrices
    svals = [np.array(init_svds[b][1]) for b in range(n_blocks)]
    trans_svals = get_k_mcca_block_gram_svals(
        svals=svals, regs=regs, diag_mode=diag_mode, n_samples=n_samples
    )

    # constructe matrix for eigen decomposition
    C = [[None for b in range(n_blocks)] for b in range(n_blocks)]
    for b in range(n_blocks):
        C[b][b] = np.eye(n_features_reduced[b])

    for (a, b) in combinations(range(n_blocks), 2):

        U_a = reduced[a]
        s_a = svals[a]
        t_a = trans_svals[a]
        q_a = s_a * (1 / np.sqrt(t_a))

        U_b = reduced[b]
        s_b = svals[b]
        t_b = trans_svals[b]
        q_b = s_b * (1 / np.sqrt(t_b))

        C[a][b] = (U_a * q_a).T @ (U_b * q_b)
        C[b][a] = C[a][b].T

    C = np.block(C)

    evals, evecs_red = eigh_wrapper(A=C, rank=n_components)
    evecs_red = split(evecs_red, dims=n_features_reduced, axis=0)

    gevecs = [None for b in range(n_blocks)]
    for b in range(n_blocks):

        U = reduced[b]
        t = trans_svals[b]
        ev_red = evecs_red[b]

        # ev = U @ ev_red
        # gevecs[b] = (U * (1 / np.sqrt(t))) @ (U.T @ ev)
        gevecs[b] = (U * (1 / np.sqrt(t))) @ ev_red

    block_scores = [Ks[b] @ gevecs[b] for b in range(n_blocks)]
    common_norm_scores, col_norms = normalize_cols(sum(bs for bs in block_scores))

    # enforce deterministic output due to possible sign flips
    common_norm_scores, block_scores, gevecs = mcca_det_output(
        common_norm_scores, block_scores, gevecs
    )

    return {
        "block_scores": block_scores,
        "dual_vars": gevecs,
        "common_norm_scores": common_norm_scores,
        "cs_col_norms": col_norms,
        "evals": evals,
        "centerers": centerers,
        "init_svds": init_svds,
    }


k_mcca_evp_svd.__doc__ = dedent(
    """
    Computes kernel MCCA using the eigenvector formulation where we compute matrix square roots via SVDs. By throwing out zero singular values of the kernel blocks we can avoid singularity issues.

    Parameters
    ----------
    {Ks}

    {basic}

    {sval_killing}

    precomp_svds: None, list of tuples
        Precomputed SVDs of each blocks kernel matrix.

    {diag_mode}

    Output
    ------
    {score_out}

    dual_vars: list of array-like
        Dual variables for projections.

    {centerers}

    init_svds: list of tuples
        SVDs of the kernel matrices.

    """
).format(**_kmcca_docs)


def get_k_mcca_block_gram_svals(svals, regs=None, diag_mode="A", n_samples=None):

    n_blocks = len(svals)
    regs = check_regs(regs=regs, n_blocks=n_blocks)

    transf_svals = [None for b in range(n_blocks)]
    for b in range(n_blocks):
        s = np.array(svals[b])

        if regs is None:
            r = None
        else:
            r = regs[b]

        if r is None or r == 0:
            t = s ** 2

        elif diag_mode == "A":
            t = (1 - r) * s ** 2 + r * s

        elif diag_mode == "B":
            assert n_samples is not None
            if np.isclose(r, 1):
                t = s
            else:
                kappa = r / (1 - r)
                t = (1 - r) * (s + 0.5 * n_samples * kappa) ** 2

        elif diag_mode == "C":
            t = (1 - r) * s ** 2 + r

        transf_svals[b] = t

    return transf_svals


get_k_mcca_block_gram_svals.__doc__ = dedent(
    """
    Gets the singular values of the block gram matrices for the various options.

    Parameters
    ----------
    {Ks}

    svals: list of array-like
        Singular values of each kernel matrix.

    regs: None, float, List
        Regulariation for each block.

    {diag_mode}

    n_samples: None, int
        Number of samples. Needed for mode B.

    Output
    ------
    transf_svals: list of array-like
        Singular values of each block's gram matrix for the given diag_mode.

    """
).format(**_kmcca_docs)
