import numpy as np
import matplotlib.pyplot as plt


def plot_joint_diagnostic(
    joint_svals,
    wedin_sv_samples,
    random_sv_samples,
    min_signal_rank,
    wedin_percentile=5,
    random_percentile=95,
    fontsize=20,
):

    fontsize_large = fontsize
    fontsize_small = int(fontsize_large * 0.75)

    # compute sv_threshold
    wedin_cutoff = np.percentile(wedin_sv_samples, wedin_percentile)
    rand_cutoff = np.percentile(random_sv_samples, random_percentile)
    svsq_cutoff = max(rand_cutoff, wedin_cutoff)
    joint_rank_est = sum(joint_svals ** 2 > svsq_cutoff)

    wedin_low = np.percentile(wedin_sv_samples, 100 - wedin_percentile)
    wedin_high = np.percentile(wedin_sv_samples, wedin_percentile)

    rand_low = np.percentile(random_sv_samples, random_percentile)
    rand_high = np.percentile(random_sv_samples, 100 - random_percentile)

    if rand_cutoff > wedin_cutoff:
        rand_lw = 4
        wedin_lw = 2

    else:
        rand_lw = 2
        wedin_lw = 4

    rand_label = "random {:d}th percentile ({:1.3f})".format(
        random_percentile, rand_cutoff
    )
    wedin_label = "wedin {:d}th percentile ({:1.3f})".format(
        wedin_percentile, wedin_cutoff
    )

    # wedin cutoff
    plt.axvspan(wedin_low, wedin_high, alpha=0.1, color="blue")
    plt.axvline(
        wedin_cutoff, color="blue", ls="dashed", lw=wedin_lw, label=wedin_label
    )

    # random cutoff
    plt.axvspan(rand_low, rand_high, alpha=0.1, color="red")
    plt.axvline(
        rand_cutoff, color="red", ls="dashed", lw=rand_lw, label=rand_label
    )

    # plot joint singular values
    first_joint = True
    first_nonjoint = True
    svals = joint_svals[0:min_signal_rank]
    for i, sv in enumerate(svals):
        sv_sq = sv ** 2

        if sv_sq > svsq_cutoff:

            label = "joint singular value" if first_joint else ""
            first_joint = False

            color = "black"
        else:

            label = "nonjoint singular value" if first_nonjoint else ""
            first_nonjoint = False

            color = "grey"

        plt.axvline(
            sv_sq,
            ymin=0.05,
            ymax=0.95,
            color=color,
            label=label,
            lw=2,
            zorder=2,
        )

        spread = 0.05 * (max(svals) - min(svals))
        plt.hlines(
            y=1 - (0.25 + 0.25 * i / min_signal_rank),
            xmin=sv_sq - spread,
            xmax=sv_sq + spread,
            color=color,
        )

    plt.xlabel("squared singular value", fontsize=fontsize_large)
    plt.legend(fontsize=fontsize_small)
    plt.ylim([0, 1])
    plt.xlim(xmin=1)
    plt.title(
        "joint singular value thresholding"
        " (joint rank estimate = {:d})".format(joint_rank_est),
        fontsize=fontsize_large,
    )

    # format axes
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axes.get_yaxis().set_ticks([])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(fontsize_small)
