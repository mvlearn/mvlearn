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
   tutorials/embed/linear_kcca_validation_tutorial
   tutorials/embed/mvmds_tutorial
   tutorials/embed/Omnibus Embedding for Multiview Data
   tutorials/embed/SplitAE Tutorial
   tutorials/embed/SplitAE Simulated Data
   tutorials/embed/pls_tutorial
   tutorials/embed/pls_simulation
   
Test Dataset
============
In order to conviently run tools in this package on multview data, data from the publicly available  External hyperlinks, like `UCI multiple features dataset <https://archive.ics.uci.edu/ml/datasets/Multiple+Features>`_ are provided with a dataloader to make access simple.

.. toctree::
   :maxdepth: 1

   tutorials/datasets/load_UCImultifeature_data
