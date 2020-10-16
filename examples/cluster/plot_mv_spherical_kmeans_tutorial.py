"""
===================================
Multiview Spherical KMeans Tutorial
===================================

This tutorial demonstrates the multiview spherical k-means algorithm
on 2 views of the UCI multiview digits dataset.

"""


from mvlearn.datasets import load_UCImultifeature
from mvlearn.cluster import MultiviewSphericalKMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score as nmi_score
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter('ignore')  # Ignore warnings

# Load in UCI digits multiple feature dataset as an example

RANDOM_SEED = 5
# Load dataset along with labels for digits 0 through 4
n_class = 5
Xs, labels = load_UCImultifeature(
    select_labeled=list(range(n_class)), views=[0, 1])


# Creating a function to display data and the results of clustering
def display_plots(pre_title, data, labels):
    # plot the views
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    dot_size = 10
    ax[0].scatter(data[0][:, 0], data[0][:, 1], c=labels, s=dot_size)
    ax[0].set_title(pre_title + ' View 1')
    ax[0].axes.get_xaxis().set_visible(False)
    ax[0].axes.get_yaxis().set_visible(False)

    ax[1].scatter(data[1][:, 0], data[1][:, 1], c=labels, s=dot_size)
    ax[1].set_title(pre_title + ' View 2')
    ax[1].axes.get_xaxis().set_visible(False)
    ax[1].axes.get_yaxis().set_visible(False)

    plt.show()


###############################################################################
# Multiview spherical KMeans clustering on 2 views
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Here we will demonstrate the performance of the multiview spherical kmeans
# clustering. We will evaluate the purity of the resulting clusters with
# respect to the class labels using the normalized mutual information metric.
#
# Use the MultiviewSphericalKMeans instance to cluster the data
m_kmeans = MultiviewSphericalKMeans(
    n_clusters=n_class, random_state=RANDOM_SEED)

m_clusters = m_kmeans.fit_predict(Xs)

# Compute nmi between true class labels and multiview cluster labels
m_nmi = nmi_score(labels, m_clusters)
print('multiview NMI Score: {0:.3f}\n'.format(m_nmi))

###############################################################################
# Multiview spectral clustering results and the true clusters
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We will display the clustering results of the multiview kmeans clustering
# algorithm below, along with the true class labels.


# Running TSNE to display clustering results via low dimensional embedding
tsne = TSNE()
new_data_1 = tsne.fit_transform(Xs[0])
new_data_2 = tsne.fit_transform(Xs[1])
new_data = [new_data_1, new_data_2]

display_plots('True Labels', new_data, labels)
display_plots('multiview KMeans Clusters', new_data, m_clusters)

###############################################################################
# Multiview spherical KMeans clustering different parameters
# ----------------------------------------------------------
# Here we will follow a similar procedure as before, but we will be using a
# different configuration of parameters for multiview Spherical KMeans
# Clustering.

# Use the MultiviewSphericalKMeans instance to cluster the data
m_kmeans = MultiviewSphericalKMeans(
    n_clusters=n_class, n_init=10, max_iter=6, patience=2,
    random_state=RANDOM_SEED)
m_clusters = m_kmeans.fit_predict(Xs)

# Compute nmi between true class labels and multiview cluster labels
m_nmi = nmi_score(labels, m_clusters)
print('multiview NMI Score: {0:.3f}\n'.format(m_nmi))
