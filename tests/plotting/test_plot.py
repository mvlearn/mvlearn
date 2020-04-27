import pytest
from mvlearn.plotting import crossviews_plot
from mvlearn.plotting import quick_visualize
import matplotlib.pyplot as plt

Xs = [[[1, 0], [0, 1]], [[1, 0], [0, -1]]]
labels = [1,2]

def test_crossview_default():
    crossviews_plot(Xs)

def test_crossview_params():
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

def test_crossview_wrong_dimensions():
    with pytest.raises(ValueError):
        crossviews_plot(Xs, dimensions=[0, 2])
    with pytest.raises(ValueError):
        crossviews_plot(Xs, dimensions=2)

def test_quick_vis_default():
    quick_visualize(Xs)

def test_quick_vis_params():
    _ = quick_visualize(
        Xs,
        labels=labels,
        title="Test",
        cmap="RdBu",
        context=None,
        show=False,
        ax_ticks=False,
        ax_labels=False,
    )
