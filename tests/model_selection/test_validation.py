import numpy as np

from sklearn.model_selection import cross_validate as sk_cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils.fixes import loguniform

from mvlearn.model_selection import cross_validate
from mvlearn.compose import ConcatMerger
from mvlearn.model_selection._search import GridSearchCV, RandomizedSearchCV
from mvlearn.embed import MCCA


def test_cross_validate():
    n_samples = 100
    n_features = [2, 3, 4]
    rng = np.random.RandomState(0)
    Xs = [rng.randn(n_samples, n_feature) for n_feature in n_features]
    X = np.hstack(Xs)
    y = rng.randint(2, size=n_samples)
    mvpipe = Pipeline([('merge', ConcatMerger()),
                       ('logreg', LogisticRegression(random_state=rng))])
    estimator = LogisticRegression(random_state=rng)
    # Check that cv on mvpipe and estimator gives same result
    mvscores = cross_validate(mvpipe, Xs, y)
    scores = sk_cross_validate(estimator, X, y)
    assert (scores['test_score'] == mvscores['test_score']).all()


def test_gridsearchcv():
    n_samples = 100
    n_features = [2, 3, 4]
    rng = np.random.RandomState(0)
    Xs = [rng.randn(n_samples, n_feature) for n_feature in n_features]
    param_grid = {
        'regs': [[0.5, 0.4, 0.3], [1, 2, 3], 0.1],
    }
    model = MCCA()

    def scorer(estimator, X, y):
        scores = estimator.score(X)
        return np.mean(scores)

    cv_model = GridSearchCV(model, cv=5, param_grid=param_grid, scoring=scorer).fit(Xs)


def test_randomizedsearchcv():
    n_samples = 100
    n_features = [2, 3, 4]
    n_iter_search=10
    rng = np.random.RandomState(0)
    Xs = [rng.randn(n_samples, n_feature) for n_feature in n_features]
    param_dists = {
        'regs': [loguniform(1e-4, 1e0), loguniform(1e-4, 1e0), [0.1]],
    }
    model = MCCA()

    def scorer(estimator, X, y):
        scores = estimator.score(X)
        return np.mean(scores)

    cv_model = RandomizedSearchCV(model, cv=5, param_distributions=param_dists, scoring=scorer,n_iter=n_iter_search).fit(Xs)

