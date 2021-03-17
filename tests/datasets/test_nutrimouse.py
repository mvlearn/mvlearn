import numbers
import numpy as np
from mvlearn.datasets import load_nutrimouse


Xs_filenames = ["gene", "lipid"]
y_filenames = ["genotype", "diet"]


def test_return_Xs_y():
    Xs_y = load_nutrimouse(return_Xs_y=True)
    Xs, y = Xs_y
    assert Xs[0].shape == (40, 120)
    assert Xs[1].shape == (40, 21)
    assert y.shape == (40, 2)
    assert len(np.unique(y[:, 0])) == 2
    assert len(np.unique(y[:, 1])) == 5
    assert issubclass(y.dtype.type, numbers.Integral)


def test_data_dict():
    data = load_nutrimouse(return_Xs_y=False)
    for key, n_features in zip(Xs_filenames, (120, 21)):
        assert data[key].shape == (40, n_features)

    for key, n_unique in zip(y_filenames, (2, 5)):
        assert data[key].shape == (40,)
        assert len(np.unique(data[key])) == n_unique
        assert issubclass(data[key].dtype.type, numbers.Integral)

    for key, n_unique in zip(Xs_filenames, (120, 21)):
        assert len(data[key + '_feature_names']) == n_unique

    for key, n_unique in zip(y_filenames, (2, 5)):
        assert len(data[key + '_names']) == n_unique
