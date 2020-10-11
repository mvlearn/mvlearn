"""
=====================================================
Loading and Viewing the UCI Multiple Features Dataset
=====================================================

"""

from mvlearn.datasets import load_UCImultifeature
from mvlearn.plotting import quick_visualize

###############################################################################
# Load the data and labels
# ^^^^^^^^^^^^^^^^^^^^^^^^
# We can either load the entire dataset (all 10 digits) or select certain
# digits. Then, visualize in 2D.

# Load entire dataset
full_data, full_labels = load_UCImultifeature()

print("Full Dataset\n")
print("Views = " + str(len(full_data)))
print("First view shape = " + str(full_data[0].shape))
print("Labels shape = " + str(full_labels.shape))

quick_visualize(full_data, labels=full_labels, title="10-class data")

###############################################################################
# Load only 2 classes of the data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Also, shuffle the data and set the seed for reproducibility. Then, visualize
# in 2D.

# Load only the examples labeled 0 or 1, and shuffle them,
# but set the random_state for reproducibility
partial_data, partial_labels = load_UCImultifeature(
    select_labeled=[0, 1], shuffle=True, random_state=42)

print("\n\nPartial Dataset (only 0's and 1's)\n")
print("Views = " + str(len(partial_data)))
print("First view shape = " + str(partial_data[0].shape))
print("Labels shape = " + str(partial_labels.shape))

quick_visualize(partial_data, labels=partial_labels, title="2-class data")
