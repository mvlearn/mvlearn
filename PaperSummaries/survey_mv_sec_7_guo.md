# Summary: Survey of Multiview Learning  
https://arxiv.org/pdf/1304.5634.pdf

## 1 Introduction

- In most sciences, data can be collected from diverse domains with different properties
- The variables from each "data example" can be naturally partitioned into **views** or groups
- Conventional machine learning algorithms concatenate all views into single view (single-view learning)
  - This can lead to overfitting in small sample spaces (why?)
  - Not physically meaningful since each view has different statistical properties

- Introducing multiview learning:
  - One function to model a particular view
  - Jointly optimize all functions to exploit redundant views of same input data and improve learning performance
  - Three main groups of algorithms: **co-training, multiple kernel learning, subspace learning**

- Cotraining (https://dl.acm.org/citation.cfm?id=279962)
  - Maximize mutual agreement on two distinct views of unlabled data
  - Different variants:
    - Generalized EM (http://www.kamalnigam.com/papers/cotrain-CIKM00.pdf)
    - Active Learning with co-training for robust semi-supervised learning (http://usc-isi-i2.github.io/papers/muslea02-icml-robust.pdf)
    - Many others
  - Three really important assumptions:
    - **sufficiency**: each view is sufficient for classification on its own
    - **compatibility**: target function of both views predict same labels for co-occurring features with high-probability
      - Another way of phrasing this: functions learned on different views give similar predictions
    - **conditional independence** (critical): views are conditionally independent given label
      - too strong to satisfy in practice

- Multiple Kernel Learning (MKL)
  - Quick review: **kernel** is a similarity function over pairs of data points (think dot product)
  - Originally designed to control search space of possible kernel matrices
  - Different kernels naturally represent different views
  - Combining kernels improves learning performance (wat why?)
  - Semi-definite programming problem! (approximation algorithms!)
  - Lots of good theory supporting it

- Subspace Learning
  - Learn a latent subspace shared by multiple views by assuming input views are generated from this latent subspace
  - Dimensionality of latent subspace must be smaller than generated views (reduces curse of dimensionality)
  - Performance classification/clustering on this latent subspace!
  - **Canonical correlation analysis (CCA)** and **Kernel CCA (KCCA)**
    - Explore basis vectors for two sets of variables by mutually maximizing the correlations between the projections onto the basis vectors (what?)

- Other related fields:
  - **Active learning**: minimize amount of labeled data required for learning a concept
  - **Ensemble learning**: use multiple learners and combine their predictions
    - bagging vs cotraining
  - **Domain Adaptation** (this sounds like transfer learning?): adapt a prediction model trained on data from source domain to different domain


## 2 Principles for Multi-view Learning

- Multiview learning is different because it demands redundant views of the same input data
- This means learning can be done with abundant information
- However, if learning method cannot cope with multiple views, performance is degraded
- Two major principles that ensure the success of multiview learning algorithms: **consensus** and **complementary**

- Consensus Principle
  - Maximize the agreement on multiple views
  - Minimizing disagreement rate of the two hypothesis minimizes the error rate of each hypothesis! (See paper for important inequality)
  - Used in a lot of algorithms even if people are unaware of it

- Complementary Principle
  - Each view of data may contain some knowledge that other views do not have
  - Multiple views can be used to comprehensively describe the data


## 7 Subspace Learning
Review: Subspace learning aims to obtain a latent subspace from multiple views under the assumption that all input views are generated from this latent space.

- Canonical Correlation Analysis:
  - For two views *X* and *Y*, CCA computes two projection vectors (w<sub>x</sub>, w<sub>y</sub>)(such that a correlation coefficient between the two projection vectors is maximized)
  - w<sub>x</sub> lives in same space as *X*. Same for w<sub>y</sub>
  - Can be solved as an optimization problem
  - linear feature extraction algorithm

- Kernel CCA:
  - In real world datasets, linear projections cannot capture the properties of the data
  - Kernel methods map the data to a higher dimensional space and then apply linear methods in that space
  - Use dual formulation of optimization problem for CCA -> the kernel matrices are captured

- Uses of CCA:
  - CCA is most commonly used as a general multiview dimensionality reduction algorithm
  - Single view approach with supervised learning; run CCA on data and labels to project data into lower-dimensional space directed by class information
  - Supervised extension of CCA -> Generalized Multi-view Analysis
  - Use CCA to project to subspace spanned by the means (?) leads to good clustering results
  - Laplacian regularization idea to deal with data examples with missing features
  - CCA also has multiple kernel learning uses

- Multi-view Fisher Discriminant Analysis:
  - CCA ignores label information
  - https://www.researchgate.net/publication/253592929_Multiview_Fisher_Discriminant_Analysis
  - Diethe et. al generalized Fisher's Discriminant Analysis for multiview data in a supervised setting
  - Surprise: it's another optimization problem
  - There's a kernel version as well

- Multi-view Embedding:
  - High dimensionality may lead to high variance, noise, over-fitting, and higher complexity/inefficiency in learners
  - Therefore, it is necessary to conduct dimensionality reduction and generate low-dimensional representations of these features
  - Dimensional reduction for each feature is not ideal considering underlying connections between them (how do you reduce just one feature? am confused by this)
    - I think this means choosing which features give the most information one by one
  - Therefore, you want to embed all features simultaneously and output a meaningful low dimensional embedding shared by all features
  - Multi-view spectral embedding (the math and explanations got insane here... need to reread)

- Multi-view Metric Learning
  - Construct embedding projections from data in different representations into shared feature space so Euclidean distance in this space is meaningful within a view and between different views
  - There's some crazy convex optimization problem that you can solve efficiently in a semi-supervised setting

- Latent Space Models:
  - Above methods focus on meaningful dimensionality reduction for multi-view data
  - Other methods focus on analyzing relationships between different views
  - These methods can build latent space models:
    - a couple subtopics to explore: shared gaussian process latent variable model, shared kernel information embedding, factorized orthogonal latent space (with or without structured data), latent space markov model
