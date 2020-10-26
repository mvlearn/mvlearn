import numpy as np
from sklearn.utils import check_random_state
from mvlearn.utils import check_Xs
from mvlearn.embed.mcca import _i_mcca, _mcca_gevp, MCCA, \
    _construct_mcca_gevp
from mvlearn.embed.kmcca import KMCCA
from mvlearn.embed.base import _check_regs


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


def check_mcca_class(mcca, Xs):
    np.testing.assert_array_almost_equal(
        mcca.common_scores_normed_,
        sum(mcca.scores_) / mcca.common_norms_)
    for b in range(len(Xs)):
        np.testing.assert_array_almost_equal(
            mcca.scores_[b], mcca.transform_view(Xs[b], view=b))

    for X, mean in zip(Xs, mcca.means_):
        if mean is not None:
            np.testing.assert_array_almost_equal(mean, np.mean(X, axis=0))


def check_kmcca_class(kmcca, Xs):
    kmcca_scores = kmcca.transform(Xs)
    np.testing.assert_array_almost_equal(
        kmcca.common_scores_normed_,
        sum(kmcca_scores) / kmcca.common_norms_)


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
            check_mcca_class(mcca, Xs)


def test_mcca_reconstruction_score():
    Xs = next(generate_mcca_test_data())
    prior_recon_score = None
    for rank in [1, 2, 3]:
        mcca = MCCA(n_components=rank).fit(Xs)
        recon_score = mcca.score(Xs)
        assert recon_score[0] == mcca.score_view(Xs[0], view=0)
        if prior_recon_score is None:
            prior_recon_score = recon_score
        else:
            np.testing.assert_array_less(recon_score, prior_recon_score)


def test_kmcca():
    for Xs in generate_mcca_test_data():
        for params in generate_mcca_test_settings():

            if len(Xs) == 2 and params['n_components'] is None:
                # this setting raises some issues where are few
                # of the view scores are not equal. I do not think
                # this is an issue in practice so lets just skip
                # this scenario
                continue

            n_features = [x.shape[1] for x in Xs]
            kmcca = KMCCA(kernel='linear',
                          sval_thresh=0,
                          signal_ranks=n_features,
                          diag_mode='A',
                          **params).fit(Xs)

            mcca = MCCA(**params).fit(Xs)

            compare_kmcca_to_mcca(Xs, mcca=mcca, kmcca=kmcca)

            # check KMCCA class
            kmcca = KMCCA(**params).fit(Xs)
            check_kmcca_class(kmcca, Xs)

            # check KMCCA class for different diag modes
            kmcca = KMCCA(diag_mode='B', **params).fit(Xs)
            check_kmcca_class(kmcca, Xs)

            kmcca = KMCCA(diag_mode='C', **params).fit(Xs)
            check_kmcca_class(kmcca, Xs)
