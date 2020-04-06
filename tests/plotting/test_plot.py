import pytest
from mvlearn.plotting.plot import crossviews_plot
import matplotlib.pyplot as plt

Xs = [[[1, 0], [0, 1]], [[1, 0], [0, -1]]]
labels = [1,2]

def test_default():
    crossviews_plot(Xs)

def test_params():
    _ = crossviews_plot(
        Xs,
        labels=labels,
        dimensions=[0, 1],
        title="Test",
        cmap="RdBu",
        context=None,
        show=False,
        ax_ticks=False,
        ax_labels=False,
        equal_axes=True,
    )

def test_wrong_dimensions():
    with pytest.raises(ValueError):
        crossviews_plot(Xs, dimensions=[0, 2])
    with pytest.raises(ValueError):
        crossviews_plot(Xs, dimensions=2)

