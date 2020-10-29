# License: MIT

import matplotlib.pyplot as plt
import seaborn as sns
from ..utils.utils import check_Xs
from ..embed import MVMDS
import numpy as np


def crossviews_plot(
    Xs,
    labels=None,
    dimensions=None,
    figsize=(10, 10),
    title=None,
    cmap=None,
    show=True,
    context="notebook",
    equal_axes=False,
    ax_ticks=True,
    ax_labels=True,
    scatter_kwargs={},
    fig_kwargs={},
):
    r"""
    Plots each dimension fron one view against each dimension from a second
    view. If both views are the same, this reduces to a pairplot.

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        The two views to plot against one another. If one view has fewer
        dimensions than the other, only that many will be plotted.
    labels : boolean, default=None
        Sets the labels of the samples.
    dimensions : array-like of ints, default=None
        The dimensions of the views to plot. If `None`, all dimensions up
        to the minimum between the views will be plotted.
    figsize : tuple, default=(10,10)
        Sets the grid figure size.
    title : string, default=None
        Sets the title of the grid.
    cmap : String, default=None
        Colormap argument for matplotlib.pyplot.scatter.
    show : boolean, default=False
        Shows the plots if true. Returns the objects otherwise.
    context : one of {'paper', 'notebook', 'talk', 'poster, None},
        default='notebook'
        Sets the seaborn plotting context.
    equal_axes : boolean, default=False
        Equalizes the axes of the plots on the diagonals if true.
    ax_ticks : boolean, default=True
        Whether to have tick marks on the axes.
    ax_labels : boolean, default=True
        Whether to label the axes with the view and dimension numbers.
    scatter_kwargs : dict, default={}
        Additional matplotlib.pyplot.scatter arguments.
    fig_kwargs : dict, default={}
        Additional matplotlib.pyplot.subplots arguments.

    Returns
    -------
    (fig, axes) : tuple of the figure and its axes.
        Only returned if `show=False`.

    Notes
    -----
    Below is an example figure generated from 2 views with 2 features
    each.

    .. figure:: /figures/crossviews_plot_example.png
        :width: 250px
        :alt: Quick Visualization of Multi-view Data
        :align: center

    """
    Xs = check_Xs(Xs)
    if dimensions is None:
        n = min(Xs[0].shape[1], Xs[1].shape[1])
        dimensions = list(range(n))
    else:
        if not isinstance(dimensions, (np.ndarray, list)):
            msg = "`dimensions` must be of type list or np.ndarray"
            raise ValueError(msg)
        elif min(dimensions) < 0 or max(dimensions) >= max(
            Xs[0].shape[1], Xs[1].shape[1]
        ):
            msg = "max or min of `dimensions` is too extreme."
            raise ValueError(msg)
        n = len(dimensions)

    fig, axes = plt.subplots(n, n, figsize=figsize, **fig_kwargs)
    sns.set_context(context)
    if n == 1:
        axes = np.asarray([axes])
    for i, ax in enumerate(axes.flatten()):
        dim2 = dimensions[int(i / n)]
        dim1 = dimensions[i % n]
        if labels is None:
            ax.scatter(
                Xs[0][:, dim1], Xs[1][:, dim2], cmap=cmap, **scatter_kwargs
            )
        else:
            ax.scatter(
                Xs[0][:, dim1],
                Xs[1][:, dim2],
                cmap=cmap,
                c=labels,
                **scatter_kwargs,
            )
        if dim2 == n - 1 and ax_labels:
            ax.set_xlabel(f"View 1 Dim {dim1+1}")
        if dim1 == 0 and ax_labels:
            ax.set_ylabel(f"View 2 Dim {dim2+1}")
        if dim1 == dim2 and equal_axes:
            ax.axis("equal")
        if not ax_ticks:
            ax.set_xticks([], [])
            ax.set_yticks([], [])

    if title is not None:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(title)
    else:
        plt.tight_layout()
    if show:
        plt.show()
    else:
        return (fig, axes)


def quick_visualize(
    Xs,
    labels=None,
    figsize=(5, 5),
    title=None,
    cmap=None,
    show=True,
    context="notebook",
    ax_ticks=True,
    ax_labels=True,
    scatter_kwargs={},
    fig_kwargs={},
):
    r"""
    Computes common principal components using MVMDS for dimensionality
    reduction and plots the multi-view data on a single 2D plot for easy
    visualization. This can be thought of as the multi-view analog of
    using PCA to decompose data and plot on principal components.

    See Also
    --------
    mvlearn.embed.MVMDS

    Parameters
    ----------
    Xs : list of array-likes or numpy.ndarray
        - Xs length: n_views
        - Xs[i] shape: (n_samples, n_features_i)
        The multi-view data to reduce to a single plot.
    labels : boolean, default=None
        Sets the labels of the samples.
    figsize : tuple, default=(5,5)
        Sets the figure size.
    title : string, default=None
        Sets the title of the figure.
    cmap : String, default=None
        Colormap argument for matplotlib.pyplot.scatter.
    show : boolean, default=True
        Shows the plots if true. Returns the objects otherwise.
    context : one of {'paper', 'notebook', 'talk', 'poster, None},
        default='notebook'
        Sets the seaborn plotting context.
    ax_ticks : boolean, default=True
        Whether to have tick marks on the axes.
    ax_labels : boolean, default=True
        Whether to label the axes with the view and dimension numbers.
    scatter_kwargs : dict, default={}
        Additional matplotlib.pyplot.scatter arguments.
    fig_kwargs : dict, default={}
        Additional matplotlib.pyplot.figure arguments.

    Returns
    -------
    fig : figure object
        Only returned if `show=False`.

    Notes
    -----
    This function simply uses ``MVMDS`` with ``n_components=2`` to
    reduce arbitrarily many views of input data to 2-dimensions, then
    makes a scatter plot.

    .. figure:: /figures/quick_visualize.png
        :width: 250px
        :alt: Quick Visualization of Multi-view Data
        :align: center

    """
    Xs = check_Xs(Xs)

    mvmds = MVMDS(n_components=2)
    Xs_reduced = mvmds.fit_transform(Xs)

    fig = plt.figure(figsize=figsize, **fig_kwargs)
    sns.set_context(context)

    if labels is None:
        plt.scatter(
            Xs_reduced[:, 0], Xs_reduced[:, 1],
            cmap=cmap, **scatter_kwargs
        )
    else:
        plt.scatter(
            Xs_reduced[:, 0], Xs_reduced[:, 1],
            cmap=cmap,
            c=labels,
            **scatter_kwargs,
        )
    if ax_labels:
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
    if not ax_ticks:
        plt.xticks([], [])
        plt.yticks([], [])

    if title is not None:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title(title)
    else:
        plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig
