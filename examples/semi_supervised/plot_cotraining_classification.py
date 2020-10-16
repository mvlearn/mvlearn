"""
=====================================
2-View Semi-Supervised Classification
=====================================

In this tutorial, we use the co-training classifier to do semi-supervised
binary classification on a 2-view dataset. Only 2% of the data is labeled,
but by using semi-supervised co-training, we still achieve good accuracy
and we do much better than using single-view methods trained on only the
labeled samples.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

from mvlearn.semi_supervised import CTClassifier
from mvlearn.datasets import load_UCImultifeature

###############################################################################
# Load the UCI Multiple Digit Features Dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To simulate a semi-supervised learning scenario, randomly remove 98% of the
# labels.

data, labels = load_UCImultifeature(select_labeled=[0, 1])

# Use only the first 2 views as an example
View0, View1 = data[0], data[1]

# Split both views into testing and training
View0_train, View0_test, labels_train, labels_test = train_test_split(
    View0, labels, test_size=0.33, random_state=42)
View1_train, View1_test, labels_train, labels_test = train_test_split(
    View1, labels, test_size=0.33, random_state=42)

# Randomly remove all but 4 of the labels
np.random.seed(6)
remove_idx = np.random.rand(len(labels_train),) < 0.98
labels_train[remove_idx] = np.nan
not_removed = np.where(~remove_idx)
print("Remaining labeled sample labels: " + str(labels_train[not_removed]))

###############################################################################
# Co-Training on 2 Views vs. Single View Semi-Supervised Learning
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Here, we use the default co-training classifier, which uses Gaussian naive
# bayes classifiers for both views. We compare its performance to the single-
# view semi-supervised setting with the same basic classifiers, and with the
# naive technique of concatenating the two views and performing single view
# learning.

# Single view semi-supervised learning
gnb0 = GaussianNB()
gnb1 = GaussianNB()
gnb2 = GaussianNB()

# Train on only the examples with labels
gnb0.fit(View0_train[not_removed, :].squeeze(), labels_train[not_removed])
y_pred0 = gnb0.predict(View0_test)
gnb1.fit(View1_train[not_removed, :].squeeze(), labels_train[not_removed])
y_pred1 = gnb1.predict(View1_test)
# Concatenate the 2 views for naive "multiview" learning
View01_train = np.hstack(
    (View0_train[not_removed, :].squeeze(),
     View1_train[not_removed, :].squeeze()))
View01_test = np.hstack((View0_test, View1_test))
gnb2.fit(View01_train, labels_train[not_removed])
y_pred2 = gnb2.predict(View01_test)

print("Single View Accuracy on First View: {0:.3f}\n".format(
    accuracy_score(labels_test, y_pred0)))
print("Single View Accuracy on Second View: {0:.3f}\n".format(
    accuracy_score(labels_test, y_pred1)))
print("Naive Concatenated View Accuracy: {0:.3f}\n".format(
    accuracy_score(labels_test, y_pred2)))

# Multi-view co-training semi-supervised learning
# Train a CTClassifier on all the labeled and unlabeled training data
ctc = CTClassifier()
ctc.fit([View0_train, View1_train], labels_train)
y_pred_ct = ctc.predict([View0_test, View1_test])

print(f"Co-Training Accuracy on 2 Views: \
    {accuracy_score(labels_test, y_pred_ct):.3f}")

###############################################################################
# Select a different base classifier for the CTClassifier
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Now, we use a random forest classifier with different attributes for each
# view.
# Furthermore, we manually select the number of positive (p) and negative (n)
# examples chosen each round in the co-training process, and we define the
# unlabeled pool size to draw them from and the number of iterations of
# training to perform.


# Single view semi-supervised learning
rfc0 = RandomForestClassifier(n_estimators=100, bootstrap=True)
rfc1 = RandomForestClassifier(n_estimators=6, bootstrap=False)
rfc2 = RandomForestClassifier(n_estimators=100, bootstrap=False)

# Train on only the examples with labels
rfc0.fit(View0_train[not_removed, :].squeeze(), labels_train[not_removed])
y_pred0 = rfc0.predict(View0_test)
rfc1.fit(View1_train[not_removed, :].squeeze(), labels_train[not_removed])
y_pred1 = rfc1.predict(View1_test)
# Concatenate the 2 views for naive "multiview" learning
View01_train = np.hstack(
    (View0_train[not_removed, :].squeeze(),
     View1_train[not_removed, :].squeeze()))
View01_test = np.hstack((View0_test, View1_test))
rfc2.fit(View01_train, labels_train[not_removed])
y_pred2 = rfc2.predict(View01_test)

print("Single View Accuracy on First View: {0:.3f}\n".format(
    accuracy_score(labels_test, y_pred0)))
print("Single View Accuracy on Second View: {0:.3f}\n".format(
    accuracy_score(labels_test, y_pred1)))
print("Naive Concatenated View Accuracy: {0:.3f}\n".format(
    accuracy_score(labels_test, y_pred2)))

# Multi-view co-training semi-supervised learning
rfc0 = RandomForestClassifier(n_estimators=100, bootstrap=True)
rfc1 = RandomForestClassifier(n_estimators=6, bootstrap=False)
ctc = CTClassifier(rfc0, rfc1, p=2, n=2, unlabeled_pool_size=20, num_iter=100)
ctc.fit([View0_train, View1_train], labels_train)
y_pred_ct = ctc.predict([View0_test, View1_test])

print(f"Co-Training Accuracy: \
    {accuracy_score(labels_test, y_pred_ct):.3f}")

# Get the prediction probabilities for all the examples
y_pred_proba = ctc.predict_proba([View0_test, View1_test])
print("Full y_proba shape = " + str(y_pred_proba.shape))
print("\nFirst 10 class probabilities:\n")
print(y_pred_proba[:10, :])
