*********
Tutorials
*********

Clustering
==========
The following tutorials demonstrate the effectiveness of clustering algorithms designed specifically
for multiview datasets.

.. toctree::
   :maxdepth: 1
      
   tutorials/cluster/MVKMeans/MultiviewKMeans_Tutorial
   tutorials/cluster/MVSpectralClustering/MultiviewSpectralClustering_Tutorial
   tutorials/cluster/MVSphericalKMeans/MVSphericalKMeans_Tutorial

Cotraining
==========
The following tutorials demonstrate how effectiveness of cotraining in certain multiview scenarios to 
boost accuracy over single view methods.

.. toctree::
   :maxdepth: 1
   
   tutorials/cotraining/cotraining_classification_exampleusage
   tutorials/cotraining/cotraining_classification_simulatedperformance

Embedding
=========
Inference on and visualization of multiview data often requires low-dimensional representations of the data, known as *embeddings*. Below are tutorials for computing such embeddings on multiview data.

.. toctree::
   :maxdepth: 1
   
   tutorials/embed/gcca_tutorial
   tutorials/embed/gcca_simulation
   tutorials/embed/kcca_tutorial
   tutorials/embed/dcca_tutorial
   tutorials/embed/cca_comparison
   tutorials/embed/mvmds_tutorial
   tutorials/embed/Omnibus Embedding for Multiview Data
   tutorials/embed/SplitAE Tutorial
   tutorials/embed/SplitAE Simulated Data

Plotting
========
Methods build on top of Matplotlib and Seaborn have been implemented for convenient plotting of multiview data. See examples of such plots on simulated data.

.. toctree::
   :maxdepth: 1

   tutorials/plotting/quick_visualize_tutorial
   tutorials/datasets/load_UCImultifeature
   tutorials/datasets/GaussianMixtures
   
Test Dataset
============
In order to conviently run tools in this package on multview data, data can be simulated or  be accessed from the publicly available `UCI multiple features dataset <https://archive.ics.uci.edu/ml/datasets/Multiple+Features>`_ using a dataloader in this package.

.. toctree::
   :maxdepth: 1

   tutorials/datasets/load_UCImultifeature
   tutorials/datasets/GaussianMixtures