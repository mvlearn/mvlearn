import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_block_heatmaps(blocks):
    """
    Plots a heat map of a bunch of data blocks
    """
    num_blocks = len(blocks)
    if hasattr(blocks, "keys"):
        block_names = list(blocks.keys())
    else:
        block_names = list(range(len(blocks)))

    for k, bn in enumerate(block_names):
        plt.subplot(1, num_blocks, k + 1)
        sns.heatmap(
            blocks[bn], xticklabels=False, yticklabels=False, cmap="RdBu"
        )
        plt.title("{}".format(bn))


def jive_full_estimate_heatmaps(full_block_estimates, blocks):
    """
    Plots the full JVIE estimates: X, J, I, E
    """
    num_blocks = len(full_block_estimates)

    # plt.figure(figsize=[10, num_blocks * 10])

    block_names = list(full_block_estimates.keys())
    for k, bn in enumerate(block_names):

        # grab data
        X = blocks[bn]
        J = full_block_estimates[bn]["joint"]
        I = full_block_estimates[bn]["individual"]
        E = full_block_estimates[bn]["noise"]

        # observed data
        plt.subplot(4, num_blocks, k + 1)
        sns.heatmap(X, xticklabels=False, yticklabels=False, cmap="RdBu")
        plt.title("{} observed data".format(bn))

        # full joint estimate
        plt.subplot(4, num_blocks, k + num_blocks + 1)
        sns.heatmap(J, xticklabels=False, yticklabels=False, cmap="RdBu")
        plt.title("joint")

        # full individual estimate
        plt.subplot(4, num_blocks, k + 2 * num_blocks + 1)
        sns.heatmap(I, xticklabels=False, yticklabels=False, cmap="RdBu")
        plt.title("individual")

        # full noise estimate
        plt.subplot(4, num_blocks, k + 3 * num_blocks + 1)
        sns.heatmap(E, xticklabels=False, yticklabels=False, cmap="RdBu")
        plt.title("noise ")
