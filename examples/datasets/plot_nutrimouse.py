"""
=====================================================
An mvlearn casestudy: the Nutrimouse dataset
=====================================================

In this tutorial, we show how one may utilize various tools of mvlearn. We
demonstrate applications to the Nutrimouse dataset from a nutrition study on
mice. The data measures 40 mice and has two views: expression levels of
potentially relevant genes and concentrations of certain fatty acids. Each
mouse has two labels: it's genetic type and diet.

[1] P. Martin, H. Guillou, F. Lasserre, S. Déjean, A. Lan, J-M.
    Pascussi, M. San Cristobal, P. Legrand, P. Besse, T. Pineau.
    "Novel aspects of PPARalpha-mediated regulation of lipid and
    xenobiotic metabolism revealed through a nutrigenomic study."
    Hepatology, 2007.
"""

# Authors: Ronan Perry
#
# License: MIT

###############################################################################
# Load the Nutrimouse dataset
# ---------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from mvlearn.datasets import load_nutrimouse

dataset = load_nutrimouse()
Xs = [dataset['gene'], dataset['lipid']]
y = np.vstack((dataset['genotype'], dataset['diet'])).T

print(f"Shapes of each view: {[X.shape for X in Xs]}")

###############################################################################
# Embed using MVMDS
# -----------------
#
# Multiview multi-dimensional scaling embeds multiview data into a single
# representation that captures information shared between both views.
# Embedding the two nutrimouse views, we can observe clear separation between
# the different genotypes and some of the diets too.

from mvlearn.embed import MVMDS  # noqa: E402

X_mvmds = MVMDS(n_components=2, num_iter=50).fit_transform(Xs)

diet_names = dataset['diet_names']
genotype_names = dataset['genotype_names']
plt.figure(figsize=(5, 5))
for genotype_idx, cmap in enumerate((cm.Blues, cm.Reds)):
    for diet_idx in range(5):
        X_idx = np.where((y == (genotype_idx, diet_idx)).all(axis=1))
        color = cmap((diet_idx + 1) / 6)
        label = diet_names[diet_idx] + f' ({genotype_names[genotype_idx]})'
        plt.scatter(*zip(*X_mvmds[X_idx]), color=color, label=label)

plt.xlabel('MVMDS component 1')
plt.ylabel('MVMDS component 2')
plt.title('MVMDS Embedding')
plt.legend()
plt.show()

###############################################################################
# Cluster using Multiview KMeans
# ------------------------------
#
# We can compare the estimated clusters from multiview KMeans to regular
# KMeans on each of the views. Multiview Kmeans clearly finds two clusters
# matching the two different genotype labels observed in the prior plots.

from mvlearn.cluster import MultiviewKMeans  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

Xs_labels = MultiviewKMeans(n_clusters=2, random_state=0).fit_predict(Xs)
X1_labels = KMeans(n_clusters=2, random_state=0).fit_predict(Xs[0])
X2_labels = KMeans(n_clusters=2, random_state=0).fit_predict(Xs[1])

sca_kwargs = {'alpha': 0.7, 's': 10}

f, axes = plt.subplots(1, 3, figsize=(10, 3))
axes[0].scatter(*zip(*X_mvmds), c=Xs_labels, **sca_kwargs)
axes[0].set_title('Multiview Kmeans Clusters')
axes[1].scatter(*zip(*X_mvmds), c=X1_labels, **sca_kwargs)
axes[1].set_title('View 1 Kmeans Clusters')
axes[2].scatter(*zip(*X_mvmds), c=X2_labels, **sca_kwargs)
axes[2].set_title('View 2 Kmeans Clusters')

for ax in axes:
    ax.set_xlabel('MVMDS component 1')
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].set_ylabel('MVMDS component 2')
axes[0].set_title('Multiview Kmeans Clusters')
plt.show()

###############################################################################
# Decomposition using AJIVE
# -------------------------
#
# We can also apply joint decomposition tools to find features across views
# that are jointly related. Using AJIVE, we can find genes and lipids that are
# jointly related.

from mvlearn.decomposition import AJIVE  # noqa: E402

ajive = AJIVE()
Xs_joint = ajive.fit_transform(Xs)

f, axes = plt.subplots(1, 2, figsize=(12, 4))
sort_idx = np.hstack((np.argsort(y[:20, 1]), np.argsort(y[20:, 1]) + 20))
y_ticks = [diet_names[j] + f' ({genotype_names[i]})' if idx %
           4 == 0 else '' for idx, (i, j) in enumerate(y[sort_idx])]

gene_ticks = [n if i in [31, 36, 76, 94] else '' for i,
              n in enumerate(dataset['gene_feature_names'])]
g = sns.heatmap(Xs_joint[0][sort_idx],
                yticklabels=y_ticks, cmap="RdBu_r", ax=axes[0],
                xticklabels=gene_ticks)
axes[0].set_title('Joint data: Gene expressions')

sns.heatmap(Xs_joint[1][sort_idx],
            yticklabels=False, cmap="RdBu_r", ax=axes[1],
            xticklabels=dataset['lipid_feature_names'])
axes[1].set_title('Joint data: Lipid concentrations')
plt.tight_layout()
plt.show()

###############################################################################
# Inference using regularized CCA
# -------------------------------
#
# CCA finds separate linear projections of views which are maximally
# correlated. We can so embed the data jointly and observe that the first two
# embeddings are highly correlated and capture the differences between
# genetic types. One can use this to construct a single view
# for subsequent inference, or to examine the loading weights across views.
# Because the genetic expression data has more features than samples, we need
# to use regularization so as to not to trivially overfit.

from mvlearn.plotting import crossviews_plot  # noqa: E402
from mvlearn.embed import CCA  # noqa: E402

cca = CCA(n_components=2, regs=[0.9, 0.1])
Xs_cca = cca.fit_transform(Xs)

y_labels = [diet_names[j] + f' ({genotype_names[i]})' for (i, j) in y]
crossviews_plot(Xs_cca, labels=y[:, 0], ax_ticks=False,
                title='CCA view embeddings', equal_axes=True)

print(f'Canonical correlations: {cca.canon_corrs(Xs_cca)}')
f, axes = plt.subplots(1, 2, figsize=(12, 2))
gene_ticks = [n if i in [31, 57, 76, 88] else '' for i,
              n in enumerate(dataset['gene_feature_names'])]
g = sns.heatmap(cca.loadings_[0].T,
                yticklabels=['Component 1', 'Component 2'],
                cmap="RdBu_r", ax=axes[0],
                xticklabels=gene_ticks)
g.set_xticklabels(gene_ticks, rotation=90)
axes[0].set_title('Gene expression loadings')

sns.heatmap(cca.loadings_[1].T,
            yticklabels=False, cmap="RdBu_r", ax=axes[1],
            xticklabels=dataset['lipid_feature_names'])
axes[1].set_title('Lipid concentration loadings')
plt.tight_layout()
plt.show()
