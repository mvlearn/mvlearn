import numpy as np

from sklearn.model_selection import cross_validate as sk_cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from mvlearn.model_selection import cross_validate
from mvlearn.merge import ConcatMerger


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
