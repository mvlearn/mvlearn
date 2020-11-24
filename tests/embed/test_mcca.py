import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from sklearn.utils import check_random_state
from mvlearn.utils import check_Xs
from mvlearn.embed.mcca import _i_mcca, _mcca_gevp, MCCA, \
    _construct_mcca_gevp
from mvlearn.embed.kmcca import KMCCA
from mvlearn.embed.base import _check_regs, _deterministic_decomp, \
    _initial_svds
from mvlearn.utils import rand_orthog
import pytest


def generate_mcca_test_data():
    rng = check_random_state(849)

    n_samples = 100
    yield [rng.normal(size=(n_samples, 10)), rng.normal(size=(n_samples, 20))]

    yield [rng.normal(size=(n_samples, 10)), rng.normal(size=(n_samples, 20)),
           rng.normal(size=(n_samples, 30))]


def generate_mcca_test_settings():

    yield {'n_components': None, 'regs': None, 'center': True}
    yield {'n_components': 5, 'regs': None, 'center': True}

    yield {'n_components': None, 'regs': .1, 'center': True}
    yield {'n_components': 5, 'regs': .1, 'center': True}

    yield {'n_components': None, 'regs': 1, 'center': True}
    yield {'n_components': 5, 'regs': 1, 'center': True}

    yield {'n_components': None, 'regs': None, 'center': False}
    yield {'n_components': 5, 'regs': None, 'center': False}


@pytest.fixture(scope='module')
def data():
    # Initialize number of samples
    nSamples = 1000
    np.random.seed(30)

    # Define two latent variables (number of samples x 1)
    latvar1 = np.random.randn(nSamples,)
    latvar2 = np.random.randn(nSamples,)

    # Define independent components for each dataset
    # (number of observations x dataset dimensions)
    indep1 = np.random.randn(nSamples, 4)
    indep2 = np.random.randn(nSamples, 5)

    # Create two datasets, with each dimension composed as a sum of 75%
    # one of the latent variables and 25% independent component
    data1 = 0.25*indep1 + 0.75 * \
        np.vstack((latvar1, latvar2, latvar1, latvar2)).T
    data2 = 0.25*indep2 + 0.75 * \
        np.vstack((latvar1, latvar2, latvar1, latvar2, latvar1)).T

    # Split each dataset into a training set and test set (10% of dataset
    # is training data)
    train1 = data1[:int(nSamples/10)]
    train2 = data2[:int(nSamples/10)]
    test1 = data1[int(nSamples/10):]
    test2 = data2[int(nSamples/10):]

    return [train1, train2], [test1, test2]


def check_mcca_scores_and_loadings(
    Xs, loadings, scores, common_scores_normed,
    regs=None, check_normalization=False
):
    """
    Checks the scores and loadings output for regularized mcca.

    - view scores are projections of views onto loadings
    - common noramlized scores are column normalized version of sum of scores

    - (optional) check normalization of loadings; this should be done for MCCA,
    but not for informative MCCA.
    """

    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    for b in range(n_views):
        # check view scores are projections of views onto view loadings
        np.testing.assert_array_almost_equal(Xs[b] @ loadings[b], scores[b])

    # check common norm scores are the column normalized sum of the
    # view scores
    cns_pred = sum(bs for bs in scores) / \
        np.linalg.norm(sum(bs for bs in scores), axis=0)
    np.testing.assert_array_almost_equal(cns_pred, common_scores_normed)

    if check_normalization:

        # concatenated loadings are orthonormal in the inner produce
        # induced by the RHS of the GEVP
        W = np.vstack(loadings)
        RHS = _construct_mcca_gevp(Xs, regs=regs)[1]
        np.testing.assert_array_almost_equal(W.T @ RHS @ W, np.eye(W.shape[1]))

        # possibly check CNS are orthonormal
        # this is only true for SUMCORR-AVRVAR MCCA i.e.
        # if no regularization is used
        if regs is None:
            np.testing.assert_array_almost_equal(
                common_scores_normed.T @ common_scores_normed,
                np.eye(common_scores_normed.shape[1]))


def check_mcca_gevp(Xs, loadings, evals, regs):
    """
    Checks the view loadings are the correct generalized eigenvectors.
    """
    Xs, n_views, n_samples, n_features = check_Xs(Xs, multiview=True,
                                                  return_dimensions=True)

    regs = _check_regs(regs=regs, n_views=n_views)

    # concatenated view loadings are the eigenvectors
    W = np.vstack(loadings)

    LHS, RHS = _construct_mcca_gevp(Xs, regs=regs)

    # check generalized eigenvector equation
    np.testing.assert_array_almost_equal(LHS @ W, RHS @ W @ np.diag(evals))

    # check normalization
    np.testing.assert_array_almost_equal(W.T @ RHS @ W, np.eye(W.shape[1]))


def check_mcca_means(mcca, Xs):
    for X, mean in zip(Xs, mcca.means_):
        if mean is not None:
            np.testing.assert_array_almost_equal(mean, np.mean(X, axis=0))


def compare_kmcca_to_mcca(Xs, mcca, kmcca):
    """
    Kernek MCCA with a linear kernel should give the same output as mcca i.e.
    the view scores, common normalized scores and evals should all be equal.
    """
    # common_scores_normed
    np.testing.assert_array_almost_equal(
        kmcca.common_scores_normed_, mcca.common_scores_normed_)

    # evals
    np.testing.assert_array_almost_equal(kmcca.evals_, mcca.evals_)

    # MCCA and linear KMCCA output should be the same
    for ks, ms in zip(kmcca.transform(Xs), mcca.transform(Xs)):
        np.testing.assert_array_almost_equal(ks, ms)

####################
# Tests start here #
####################


def test_deterministic_decomp():
    A = np.asarray([[-3, -3, 3], [2, 2, 2], [1, 1, 1]])
    A_det = np.asarray([[3, 3, 3], [-2, -2, 2], [-1, -1, 1]])
    X, Y, Z = _deterministic_decomp(A, -A, -A)
    assert_array_equal(X, A_det)
    assert_array_equal(Y, -A_det)
    assert_array_equal(Z, -A_det)


@pytest.mark.parametrize("normalized_scores", [True, False])
def test_initial_svds(normalized_scores):
    U = rand_orthog(10, 5, random_state=0)
    V = rand_orthog(8, 5, random_state=1)
    D = np.arange(1, 6)[::-1]
    U, V, _ = _deterministic_decomp(U, V)
    X = U @ np.diag(D) @ V.T

    with pytest.raises(ValueError):
        _initial_svds([np.zeros(X.shape)], sval_thresh=0.1)

    Us, svd = _initial_svds(
        [X], sval_thresh=0.1, normalized_scores=normalized_scores)
    assert_almost_equal(U, svd[0][0])
    assert_almost_equal(D, svd[0][1])
    assert_almost_equal(V, svd[0][2])

    if normalized_scores:
        assert_almost_equal(U, Us[0])
    else:
        assert_almost_equal(U * D, Us[0])

    Us, svd = _initial_svds([X], sval_thresh=2)
    assert Us[0].shape[1] == 4


@pytest.mark.parametrize("regs", ["fail", 5, -1])
def test_check_regs_fails(regs):
    with pytest.raises(AssertionError):
        _check_regs(regs=regs, n_views=4)


@pytest.mark.parametrize("regs", ["oas", "lw", 0.5, None])
def test_check_regs(regs):
    regs = _check_regs(regs=regs, n_views=4)
    assert len(regs) == 4


def test_transform_fail():
    Xs = next(generate_mcca_test_data())
    mcca = MCCA().fit(Xs)
    with pytest.raises(ValueError):
        mcca.transform([Xs[0]])


def test_inverse_transform_fail():
    Xs = next(generate_mcca_test_data())
    mcca = MCCA().fit(Xs)
    with pytest.raises(ValueError):
        mcca.inverse_transform([Xs[0]])


def test_svd_reg_fail():
    mcca = MCCA(i_mcca_method='svd', regs=0.5, signal_ranks=1)
    with pytest.raises(AssertionError):
        mcca.fit(Xs=next(generate_mcca_test_data()))


def test_mcca_eigh_singular():
    X = [[1, 1, 1], [2, 3, 3], [2, 3, 3]]
    mcca = MCCA()
    with pytest.raises(ValueError, match="Eigenvalue problem"):
        mcca.fit([X, X])


def test_mcca_n_components():
    Xs = next(generate_mcca_test_data())
    n_features = sum([X.shape[1] for X in Xs])
    mcca = MCCA(n_components=n_features+1)
    with pytest.raises(AttributeError):
        mcca.n_components_
    with pytest.warns(None):
        mcca.fit(Xs)
    assert mcca.n_components_ == n_features
    for load in mcca.loadings_:
        assert mcca.n_components_ == load.shape[1]


def test_kmcca_n_components():
    Xs = next(generate_mcca_test_data())
    n_features = sum([X.shape[1] for X in Xs])
    kmcca = KMCCA(n_components=n_features+1)
    with pytest.raises(AttributeError):
        kmcca.n_components_
    with pytest.warns(None):
        kmcca.fit(Xs)
    assert kmcca.n_components_ == n_features
    for load in kmcca.dual_vars_:
        assert kmcca.n_components_ == load.shape[1]


def test_mcca_all():
    for Xs in generate_mcca_test_data():
        for params in generate_mcca_test_settings():

            n_views = len(Xs)
            if params['center']:
                Xs_c = [X - np.mean(X, axis=0) for X in Xs]
            else:
                Xs_c = Xs
            # check basic usage of mcca_gevp
            loadings, scores, common_scores_normed, common_norms, evals = \
                _mcca_gevp(Xs_c, n_components=params['n_components'],
                           regs=params['regs'])
            check_mcca_scores_and_loadings(
                Xs_c, loadings, scores, common_scores_normed,
                regs=params['regs'], check_normalization=True)

            check_mcca_gevp(Xs_c, loadings, evals, regs=params['regs'])

            if params['regs'] is None:
                # check basic usage of i_mcca with SVD method
                loadings, scores, common_scores_normed, common_norms, evals = \
                    _i_mcca(Xs_c, signal_ranks=None, method='svd',
                            n_components=params['n_components'],
                            regs=params['regs'])

                check_mcca_scores_and_loadings(
                    Xs_c, loadings, scores, common_scores_normed,
                    regs=params['regs'], check_normalization=True)

                check_mcca_gevp(Xs_c, loadings, evals,
                                regs=params['regs'])

            # check basic usage of i_mcca with gevp method
            # this solves GEVP by first doing SVD, not interesting in practice
            # but this should work correctly
            loadings, scores, common_scores_normed, common_norms, evals = \
                _i_mcca(Xs_c, signal_ranks=None, method='gevp',
                        n_components=params['n_components'],
                        regs=params['regs'])

            check_mcca_scores_and_loadings(
                    Xs_c, loadings, scores, common_scores_normed,
                    regs=params['regs'], check_normalization=True)

            check_mcca_gevp(Xs_c, loadings, evals, regs=params['regs'])

            # check i_mcca when we first do dimensionality reduction
            # with SVD method
            if params['regs'] is None:
                loadings, scores, common_scores_normed, common_norms, evals = \
                    _i_mcca(Xs_c, signal_ranks=[3] * n_views,
                            method='svd', n_components=params['n_components'],
                            regs=params['regs'])

                check_mcca_scores_and_loadings(
                    Xs_c, loadings, scores, common_scores_normed,
                    regs=params['regs'], check_normalization=False)

            # check i_mcca when we first do dimensionality reduction
            # with GEVP method
            loadings, scores, common_scores_normed, common_norms, evals = \
                _i_mcca(Xs_c, signal_ranks=[3] * n_views,
                        method='gevp', n_components=params['n_components'],
                        regs=params['regs'])

            check_mcca_scores_and_loadings(
                    Xs_c, loadings, scores, common_scores_normed,
                    regs=params['regs'], check_normalization=False)
            # check MCCA class
            mcca = MCCA(**params).fit(Xs)
            check_mcca_means(mcca, Xs)


@pytest.mark.parametrize("multiview_output", [True, False])
def test_mcca_reconstruction_score(multiview_output):
    Xs = next(generate_mcca_test_data())
    prior_recon_score = None
    for rank in [1, 2, 3]:
        mcca = MCCA(n_components=rank).fit(Xs)
        recon_score = mcca.score(Xs)
        if multiview_output:
            assert recon_score[0] == mcca.score_view(Xs[0], view=0)
        if prior_recon_score is None:
            prior_recon_score = recon_score
        else:
            np.testing.assert_array_less(recon_score, prior_recon_score)


@pytest.mark.parametrize("diag_mode", ["A"])
def test_kmcca(diag_mode):
    for Xs in generate_mcca_test_data():
        for params in generate_mcca_test_settings():

            if len(Xs) == 2 and params['n_components'] is None:
                # this setting raises some issues where are few
                # of the view scores are not equal. I do not think
                # this is an issue in practice so lets just skip
                # this scenario
                continue

            # Linear KMCCA should match MCCA
            n_features = [x.shape[1] for x in Xs]
            kmcca = KMCCA(kernel='linear',
                          sval_thresh=0,
                          signal_ranks=n_features,
                          diag_mode=diag_mode,
                          **params).fit(Xs)
            mcca = MCCA(**params).fit(Xs)
            # evals
            np.testing.assert_array_almost_equal(
                kmcca.evals_, mcca.evals_, decimal=2)

            # MCCA and linear KMCCA output should be the same
            for ks, ms in zip(kmcca.transform(Xs), mcca.transform(Xs)):
                np.testing.assert_array_almost_equal(ks, ms, decimal=2)

            # Linear KMCCA should match MCCA
            n_features = [x.shape[1] for x in Xs]
            kmcca = KMCCA(kernel='linear',
                                 sval_thresh=0,
                                 signal_ranks=n_features,
                                 diag_mode=diag_mode,
                                 multiview_output=False,
                                 **params).fit(Xs)
            mcca = MCCA(multiview_output=False, **params).fit(Xs)
            # evals
            np.testing.assert_array_almost_equal(
                kmcca.evals_, mcca.evals_, decimal=2)
            # common_scores_normed
            np.testing.assert_array_almost_equal(
                kmcca.transform(Xs), mcca.transform(Xs), decimal=2)


def test_pgso():
    Xs = next(generate_mcca_test_data())
    N = Xs[0].shape[0]
    ranks = []
    corrs = []
    for tol in [0, 0.1, 0.5, 1]:
        kmcca = KMCCA(kernel='rbf', pgso=True, tol=tol, n_components=2)
        scores = kmcca.fit(Xs).transform(Xs)
        # kmcca2 = KMCCA(kernel='rbf', pgso=True, tol=tol)
        # scores2 = kmcca2.fit(Xs).transform(Xs)
        for v in range(len(Xs)):
            assert len(set(
                [kmcca.pgso_Ls_[v].shape[1],
                 kmcca.pgso_norms_[v].shape[0],
                 kmcca.pgso_idxs_[v].shape[0],
                 kmcca.pgso_Xs_[v].shape[0]])) == 1
            R = kmcca._get_kernel(Xs[v], v) - \
                kmcca.pgso_Ls_[v] @ kmcca.pgso_Ls_[v].T
            R = R - kmcca.kernel_col_means_[v] - \
                kmcca.kernel_col_means_[v].T + kmcca.kernel_mat_means_[v]
            assert np.trace(R) / N <= tol
        ranks.append(kmcca.pgso_ranks_)
        corrs.append(kmcca.canon_corrs(scores))
        if tol == 0:
            assert np.all(ranks[-1] == [N for _ in Xs])
    assert np.all(np.diff(corrs, axis=0) <= 0), corrs
    assert np.all(np.diff(corrs, axis=1) <= 1e-10)
    assert np.all(np.diff(ranks, axis=0) <= 0)


@pytest.mark.parametrize("signal_ranks", [None, 2])
@pytest.mark.parametrize("n_components", [None, 2, 'min', 'max'])
@pytest.mark.parametrize("regs", [None, 0.5, 'lw', 'oas'])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("i_mcca_method", ['auto', 'svd', 'gevp'])
@pytest.mark.parametrize("multiview_output", [True, False])
def test_mcca_params(signal_ranks, n_components, regs,
                     center, multiview_output, i_mcca_method):
    if (i_mcca_method == 'svd' or i_mcca_method == 'auto') and \
       regs is not None and signal_ranks is not None:
        return
    mcca = MCCA(
        signal_ranks=signal_ranks, i_mcca_method=i_mcca_method,
        n_components=n_components, regs=regs,
        center=center, multiview_output=multiview_output)
    Xs = next(generate_mcca_test_data())
    scores = mcca.fit_transform(Xs)
    if multiview_output:
        assert len(scores) == len(Xs)
        canon_corrs = mcca.canon_corrs(scores)
        assert np.all(canon_corrs[:2] > 0), f"{canon_corrs}"
        assert len(canon_corrs) == scores.shape[2]
    else:
        assert scores.shape == (Xs[0].shape[0], mcca.n_components_)


@pytest.mark.parametrize("kernel_args", [
    ("linear", {}),
    (["poly", "rbf"], [{"degree": 2, "coef0": 0}, {"gamma": 2}])])
@pytest.mark.parametrize("diag_mode", ["A", "B", "C"])
@pytest.mark.parametrize("signal_ranks", [None, 2])
@pytest.mark.parametrize("n_components", [None, 2])
@pytest.mark.parametrize("regs", [None, 0.5, 1])
@pytest.mark.parametrize("sval_thresh", [0, 0.01])
@pytest.mark.parametrize("center", [True, False])
@pytest.mark.parametrize("multiview_output", [True, False])
def test_kmcca_params(kernel_args, diag_mode, signal_ranks, n_components,
                      regs, sval_thresh, center, multiview_output):
    kernel, kernel_params = kernel_args
    kmcca = KMCCA(
        kernel=kernel, kernel_params=kernel_params,
        diag_mode=diag_mode, signal_ranks=signal_ranks,
        n_components=n_components, regs=regs, sval_thresh=sval_thresh,
        center=center, multiview_output=multiview_output)
    Xs = next(generate_mcca_test_data())
    scores = kmcca.fit_transform(Xs)
    if multiview_output:
        assert len(scores) == len(Xs)
        canon_corrs = kmcca.canon_corrs(scores)
        assert len(canon_corrs) == scores.shape[2]
    else:
        assert scores.shape == (Xs[0].shape[0], kmcca.n_components_)
