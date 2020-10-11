"""
================================================
Exploratory visualizations using quick_visualize
================================================

As a simple example, say we had high-dimensional multi-view data that we
wanted to quickly visualize before we begin our analysis. With
quick_visualize, we can easily do this. As an example, we will visualize the
UCI Multiple Features dataset.

"""

from mvlearn.plotting import quick_visualize
from mvlearn.datasets import load_UCImultifeature

###############################################################################
# View the UCI data
# ^^^^^^^^^^^^^^^^^
#
# Load 4-class data
Xs, y = load_UCImultifeature(select_labeled=[0, 1, 2, 3])

# Quickly visualize the data
quick_visualize(Xs, figsize=(5, 5))

###############################################################################
# Plot with class labels
# ^^^^^^^^^^^^^^^^^^^^^^

quick_visualize(Xs, labels=y, title='Labeled Classes', figsize=(5, 5))
