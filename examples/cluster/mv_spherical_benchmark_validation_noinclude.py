"""
=====================================================
Benchmarking Multiview vs. Singlview Spherical KMeans
=====================================================

In this tutorial we benchmark multiview spherical kmeans compared to
the singleview version of the same algorithm. We analyze what data
distributions lead to better performance by each model.

Note, this tutorial compares performance against the SphericalKMeans
function from the spherecluster package which is not a installed dependency
of *mvlearn*.

"""

import numpy as np
from mvlearn.cluster.mv_spherical_kmeans import MultiviewSphericalKMeans
from spherecluster import SphericalKMeans, sample_vMF
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

###############################################################################
# A function to generate 2 views of data for 2 classes
#
# This function takes parameters for means, kappas (concentration parameter),
# and number of samples for class and generates data based on those parameters.
# The underlying probability distribution of the data is a von Mises-Fisher
# distribution.


def create_data(seed, vmeans, vkappas, num_per_class=500):
    np.random.seed(seed)
    data = [[], []]
    for view in range(2):
        for comp in range(len(vmeans[0])):
            comp_samples = sample_vMF(
                vmeans[view][comp], vkappas[view][comp], num_per_class)
            data[view].append(comp_samples)
    for view in range(2):
        data[view] = np.vstack(data[view])

    labels = list()
    for ind in range(len(vmeans[0])):
        labels.append(ind * np.ones(num_per_class,))

    labels = np.concatenate(labels)

    return data, labels

###############################################################################
# Creating a function to display data and the results of clustering
#
# The following function plots both views of data given a dataset and
# corresponding labels.


def display_plots(pre_title, data, labels):
    # plot the views
    fig = plt.figure(figsize=(14, 10))
    for v in range(2):
        ax = fig.add_subplot(
            1, 2, v+1, projection='3d',
            xlim=[-1.1, 1.1], ylim=[-1.1, 1.1], zlim=[-1.1, 1.1]
        )
        ax.scatter(data[v][:, 0], data[v][:, 1], data[v][:, 2], c=labels, s=8)
        ax.set_title(pre_title + ' View ' + str(v))
        plt.axis('off')

    plt.show()

###############################################################################
# Function to perform both singleview and multiview Spherical KMeans clustering
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


def perform_clustering(seed, m_data, labels, n_clusters):
    # Singleview spherical kmeans clustering
    # Cluster each view separately
    s_kmeans = SphericalKMeans(n_clusters=n_clusters,
                               random_state=seed, n_init=100)
    s_clusters_v1 = s_kmeans.fit_predict(m_data[0])
    s_clusters_v2 = s_kmeans.fit_predict(m_data[1])

    # Concatenate the multiple views into a single view
    s_data = np.hstack(m_data)
    s_clusters = s_kmeans.fit_predict(s_data)

    # Compute nmi between true class labels and singleview cluster labels
    s_nmi_v1 = nmi_score(labels, s_clusters_v1)
    s_nmi_v2 = nmi_score(labels, s_clusters_v2)
    s_nmi = nmi_score(labels, s_clusters)
    print('Singleview View 1 NMI Score: {0:.3f}\n'.format(s_nmi_v1))
    print('Singleview View 2 NMI Score: {0:.3f}\n'.format(s_nmi_v2))
    print('Singleview Concatenated NMI Score: {0:.3f}\n'.format(s_nmi))

    # Multiview spherical kmeans clustering

    # Use the MultiviewKMeans instance to cluster the data
    m_kmeans = MultiviewSphericalKMeans(
        n_clusters=n_clusters, n_init=100, random_state=seed)
    m_clusters = m_kmeans.fit_predict(m_data)

    # Compute nmi between true class labels and multiview cluster labels
    m_nmi = nmi_score(labels, m_clusters)
    print('Multiview NMI Score: {0:.3f}\n'.format(m_nmi))

    return m_clusters

###############################################################################
# General experimentation procedures
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# For each of the experiments below, we run both singleview spherical kmeans
# clustering and multiview spherical kmeans clustering. For evaluating single-
# view performance, we run the algorithm on each view separately as well as all
# views concatenated together. We evalaute performance using normalized mutual
# information, which is a measure of cluster purity with respect to the true
# labels. For both algorithms, we use an n_init value of 100, which means that
# we run each algorithm across 100 random cluster initializations and select
# the best clustering results with respect to cluster inertia.

###############################################################################
# Performance when cluster components in both views are well separated
# --------------------------------------------------------------------
# As we can see, multiview kmeans clustering performs about as well as
# singleview spherical kmeans clustering for the concatenated views, and
# singleview spherical kmeans clustering for view 1.


RANDOM_SEED = 10

v1_kappas = [15, 15]
v2_kappas = [15, 15]
kappas = [v1_kappas, v2_kappas]
v1_mus = np.array([[-1, 1, 1], [1, 1, 1]])
v1_mus = normalize(v1_mus)
v2_mus = np.array([[1, -1, 1], [1, -1, -1]])
v2_mus = normalize(v2_mus)
v_means = [v1_mus, v2_mus]
data, labels = create_data(RANDOM_SEED, v_means, kappas)

m_clusters = perform_clustering(RANDOM_SEED, data, labels, 2)
display_plots('Ground Truth', data, labels)
display_plots('Multiview Clustering', data, m_clusters)

###############################################################################
# Performance when cluster components are relatively inseparable in both views
# ----------------------------------------------------------------------------
#
# As we can see, multiview spherical kmeans clustering performs about as
# poorly as singleview spherical kmeans clustering across both individual
# views and concatenated views as inputs.

v1_kappas = [15, 15]
v2_kappas = [15, 15]
kappas = [v1_kappas, v2_kappas]
v1_mus = np.array([[0.5, 1, 1], [1, 1, 1]])
v1_mus = normalize(v1_mus)
v2_mus = np.array([[1, -1, 1], [1, -1, 0.5]])
v2_mus = normalize(v2_mus)
v_means = [v1_mus, v2_mus]
data, labels = create_data(RANDOM_SEED, v_means, kappas)

m_clusters = perform_clustering(RANDOM_SEED, data, labels, 2)
display_plots('Ground Truth', data, labels)
display_plots('Multiview Clustering', data, m_clusters)

###############################################################################
# Performance when cluster components are somewhat separable in both views
# ------------------------------------------------------------------------
#
# Again we can see that multiview spherical kmeans clustering performs about
# as well as singleview spherical kmeans clustering for the concatenated
# views, and both of these perform better than on singleview spherical kmeans
# clustering for just one view.

v1_kappas = [15, 10]
v2_kappas = [10, 15]
kappas = [v1_kappas, v2_kappas]
v1_mus = np.array([[-0.5, 1, 1], [1, 1, 1]])
v1_mus = normalize(v1_mus)
v2_mus = np.array([[1, -1, 1], [1, -1, -0.2]])
v2_mus = normalize(v2_mus)
v_means = [v1_mus, v2_mus]
data, labels = create_data(RANDOM_SEED, v_means, kappas)

m_clusters = perform_clustering(RANDOM_SEED, data, labels, 2)
display_plots('Ground Truth', data, labels)
display_plots('Multiview Clustering', data, m_clusters)

###############################################################################
# Performance when cluster components are highly overlapping in one view
# ----------------------------------------------------------------------
#
# As we can see, multiview spherical kmeans clustering performs worse than
# singleview spherical kmeans clustering with concatenated views as inputs and
# with the best view as the input.


v1_kappas = [15, 15]
v2_kappas = [15, 15]
kappas = [v1_kappas, v2_kappas]
v1_mus = np.array([[1, -0.5, 1], [1, 1, 1]])
v1_mus = normalize(v1_mus)
v2_mus = np.array([[1, -1, 1], [1, -1, 0.6]])
v2_mus = normalize(v2_mus)
v_means = [v1_mus, v2_mus]
data, labels = create_data(RANDOM_SEED, v_means, kappas)

m_clusters = perform_clustering(RANDOM_SEED, data, labels, 2)
display_plots('Ground Truth', data, labels)
display_plots('Multiview Clustering', data, m_clusters)

###############################################################################
# Conclusions
# -----------
#
# Here, we have seen some of the limitations of multiview spherical kmeans
# clustering. From the experiments above, it is apparent that multiview
# spherical kmeans clustering performs equally as well or worse than singleview
# spherical kmeans clustering on concatenated data when views are informative
# but the data is fairly simple (i.e. only has 2 features per view). However,
# it is clear that the multiview spherical kmeans algorithm does perform better
# on well separated cluster components than it does on highly overlapping
# cluster components, which does validate it's basic functionality as a
# clustering algorithm.
