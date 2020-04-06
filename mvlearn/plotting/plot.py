# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Original work Copyright (c) 2016 Vahid Noroozi
# Modified work Copyright 2019 Zhanghao Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

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
    show : boolean, default=True
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
        ax.set_title(title)
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
    Computes common principal components and plots the multi-view data
    on a single 2D plot for easy visualization. Uses MVMDS for
    dimensionality reduction. This can be thought of as the multi-view
    analog of using PCA to decompose data and plot on principal
    components.

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
        Additional matplotlib.pyplot.subplots arguments.

    Returns
    -------
    (fig, axes) : tuple of the figure and its axes.
        Only returned if `show=False`.

    """
    Xs = check_Xs(Xs)

    mvmds = MVMDS(n_components=2)
    Xs_reduced = mvmds.fit_transform(Xs)

    fig, ax = plt.subplots(1, 1, figsize=figsize, **fig_kwargs)
    sns.set_context(context)

    if labels is None:
        ax.scatter(
            Xs_reduced[:,0], Xs_reduced[:,1],
            cmap=cmap, **scatter_kwargs
        )
    else:
        ax.scatter(
            Xs_reduced[:,0], Xs_reduced[:,1],
            cmap=cmap,
            c=labels,
            **scatter_kwargs,
        )
    if ax_labels:
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
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
