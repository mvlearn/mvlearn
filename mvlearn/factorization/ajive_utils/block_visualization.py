import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def _data_block_heatmaps(blocks):
    """
    Plots a heat map of all views
    """
    num_blocks = len(blocks)
    if hasattr(blocks, "keys"):
        block_names = list(blocks.keys())
    else:
        block_names = list(map(lambda x: x+1, list(range(len(blocks)))))

    for k, bn in enumerate(block_names):
        plt.subplot(1, num_blocks, k+1)
        sns.heatmap(
            blocks[bn-1], xticklabels=False, yticklabels=False, cmap="RdBu"
        )
        plt.title("View: {}".format(bn))


def _ajive_full_estimate_heatmaps(full_block_estimates, blocks, names=None):
    """
    Plots the full AJIVE estimates: X, J, I, E

    """
    num_blocks = len(full_block_estimates)

    if names is None:
        names = np.arange(num_blocks)

    status = isinstance(full_block_estimates, dict)

    if status:
        block_names = list(full_block_estimates.keys())

    elif not status:
        block_names = names

    for k, bn in enumerate(block_names):

        if not status:        # plotting for list
            X = blocks[k]
            J = full_block_estimates[k][0]
            I_mat = full_block_estimates[k][1]
            E = full_block_estimates[k][2]

            # observed data
            plt.subplot(4, num_blocks, k + 1)
            sns.heatmap(X, xticklabels=False, yticklabels=False, cmap="RdBu")
            plt.title("View: {} observed data".format(bn))

            # full joint estimate
            plt.subplot(4, num_blocks, k + num_blocks + 1)
            sns.heatmap(J, xticklabels=False, yticklabels=False, cmap="RdBu")
            plt.title("View: {} joint".format(bn))

            # full individual estimate
            plt.subplot(4, num_blocks, k + 2 * num_blocks + 1)
            sns.heatmap(I_mat, xticklabels=False, yticklabels=False,
                        cmap="RdBu")
            plt.title("View: {} individual".format(bn))

            # full noise estimate
            plt.subplot(4, num_blocks, k + 3 * num_blocks + 1)
            sns.heatmap(E, xticklabels=False, yticklabels=False, cmap="RdBu")
            plt.title("View: {} noise ".format(bn))

        if status:        # plotting for dict
            X = blocks[bn]
            J = full_block_estimates[bn]["joint"]
            I_mat = full_block_estimates[bn]["individual"]
            E = full_block_estimates[bn]["noise"]

            # observed data
            plt.subplot(4, num_blocks, k + 1)
            sns.heatmap(X, xticklabels=False, yticklabels=False, cmap="RdBu")
            plt.title("View: {} observed data".format(bn))

            # full joint estimate
            plt.subplot(4, num_blocks, k + num_blocks + 1)
            sns.heatmap(J, xticklabels=False, yticklabels=False, cmap="RdBu")
            plt.title("View: {} joint".format(bn))

            # full individual estimate
            plt.subplot(4, num_blocks, k + 2 * num_blocks + 1)
            sns.heatmap(I_mat, xticklabels=False, yticklabels=False,
                        cmap="RdBu")
            plt.title("View: {} individual".format(bn))

            # full noise estimate
            plt.subplot(4, num_blocks, k + 3 * num_blocks + 1)
            sns.heatmap(E, xticklabels=False, yticklabels=False, cmap="RdBu")
            plt.title("View: {} noise ".format(bn))
