import matplotlib.pyplot as plt
import seaborn as sns
from .utils import check_Xs_y
from .utils import check_Xs


def plot_2views(
    Xs,
    y=None,
    figsize=(10, 10),
    title="",
    show=True,
    context="notebook",
    equal_axes=False,
    scatter_kwargs={},
    fig_kwargs={}
):
    r"""

    """
    if y is None:
        Xs = check_Xs(Xs)
    else:
        Xs,y = check_Xs_y(Xs, y)

    n = Xs[0].shape[1]
    fig, axes = plt.subplots(n, n, figsize=figsize, **fig_kwargs)
    sns.set_context("notebook")

    for i, ax in enumerate(axes.flatten()):
        dim2 = int(i / n)
        dim1 = i % n
        if y is None:
            ax.scatter(Xs[0][:, dim1], Xs[1][:, dim2], **scatter_kwargs)
        else:
            ax.scatter(Xs[0][:, dim1], Xs[1][:, dim2], c=y, **scatter_kwargs)
        if dim2 == n - 1:
            ax.set_xlabel(f"View 1 Dim {dim1+1}")
        if dim1 == 0:
            ax.set_ylabel(f"View 2 Dim {dim2+1}")
        if dim1==dim2 and equal_axes:
            ax.axis("equal")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title)
    if show:
        plt.show()
    else:
        return (fig, axes)
