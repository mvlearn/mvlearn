import numpy as np

import pytest

from mvlearn.merge import StackTransformer, MeanTransformer


@pytest.mark.parametrize(
    "n_features",
    (
        [
            2,
        ],
        [2, 3],
        [2, 3, 4],
    ),
)
def test_stack(n_features):
    rng = np.random.RandomState(0)
    n_samples = 10
    Xs = [rng.randn(n_samples, n_feature) for n_feature in n_features]
    st = StackTransformer()
    st.fit(Xs)
    X_transformed = st.transform(Xs)
    # Check dimensions
    assert X_transformed.shape == (n_samples, sum(n_features))
    assert st.n_total_features_ == sum(n_features)
    assert st.n_views_ == len(n_features)
    assert st.n_features_ == n_features
    # Back transform
    X_init = st.inverse_transform(X_transformed)
    assert len(X_init) == len(n_features)
    assert [X.shape for X in X_init] == [
        (n_samples, n_feature) for n_feature in n_features
    ]
    for X, X2 in zip(Xs, X_init):
        assert (X == X2).all()
    # Check that you cannot transform back data of wrong dimension
    X = rng.randn(n_samples, sum(n_features) + 1)
    with np.testing.assert_raises(ValueError):
        st.inverse_transform(X)


@pytest.mark.parametrize("n_features", (2, 3))
def test_mean(n_features):
    rng = np.random.RandomState(0)
    n_views = 4
    n_samples = 10
    Xs = [rng.randn(n_samples, n_features) for _ in range(n_views)]
    mean = MeanTransformer()
    mean.fit(Xs)
    X_transformed = mean.transform(Xs)
    # Check dimensions
    assert X_transformed.shape == (n_samples, n_features)
    # Check that you cannot transform data of different number of features
    n_features = [2, 3]
    Xs = [rng.randn(n_samples, n_feature) for n_feature in n_features]
    with np.testing.assert_raises(ValueError):
        mean.fit(Xs)
