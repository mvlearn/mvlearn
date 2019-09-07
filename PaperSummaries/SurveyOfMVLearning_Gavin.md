# A Survey on Multi-view Learning
### Chang Xu, Dacheng Tao, Chao Xu

# Summary

## Introduction
3 types of MV Learning Methods
1. Co-training
    - "trains alternately to maximize the mutual agreement on two distinct views of the unlabeled data"
    - Relies on 3 assumptions
        1. sufficiency - each view is enough for classification on its own
        2. compatibility - views will predict the same labels as each other
        3. conditional independence - given the data, views are conditionally independent
2. Multiple kernel learning
    - kernels correspond to different views, which are combined linearly or nonlinearly
3. Subspace learning
    - assume input views are generated from a shared latent subspace
    - Canonical Correlation Analysis (CCA) (Hotelling, 1936)
    - Kernel CCA (KCCA) (Akaho, 2006)

## 2. Principles of Multi-view Learning
### 2.1 Consensus Principles
- We want to minimize the disagreement between the multiple hypotheses that our views have, since doing
this will minimize the error rate of each hypothesis separately
    * Many current algorithms are based on this (co-training, co-regularization, SVM-2K etc)
- "In multi-view embedding, we conduct the embedding for multiple features simultaneously
while considering the consistency and complement of different views."

### 2.2 Complementary Principle
- Different views can contain information that others do not have, so together, they describe the
data more accurately
    * Results in improved learning performance over single-view
- For example, "Nigam and Ghani (2000) used the classifier learned on one view to label the unlabeled
data, and then prepared these newly labeled examples for the next iteration of classifier
training on another view."
    * The two-view system had improved performance, since the model contained complementary information from two views
- Using multiple iterations of the same base model (but with different biases) for co-training results in the two models converging towards each other, but improved overall performance, even though no new information is being added to the data

## 3. View Generation
### 3.1 View Construction
- If data cannot be instrinsically measured in multiple ways that are different, techniques exist to sample and create different views
    * Feature set partitioning - finds disjoint subsets to construct each view
    * Random Subspace Method (RSM) - performs bootstrapping in feature space

### 3.2 View Evaluation
- View validation algorithms can predict compatibility for MV learning
- Sufficiency often doesn't hold in practice
- Inter and intra-view confidence measures exist (Liu and Yuen, 2011)

## 4. View Combination
Traditional way is to concatenate multiple views and train using single view approaches. But, this can cause overfitting and doesn't take into account the varied statistical properties of each view.
- In co-training, maximize the agreement of the predictions of the classifiers on the labeled (training) dataset, and minimize the disagreement of the predictions of the classifiers on the unlabeled (validation) dataset.
    * In each iteration, one classifier labels unlabedel data and adds it to the labeled dataset for the other.
    * Views are initially considered independently while training base learners
- For multiple kernel learning, use different kernels for each view, then combine them using a kernel based method. Views are combined just before training the full learner.
    1. Linear combination of kernels (weighted, unweighted, gated)
    2. Nonlinear combination of kernels
- Subspace learning tries to find latent subspace for all input views, similar to dimensionality reduction in a single view.
    * CCA is the multi-view correlate of PCA
        * Maximizes the correlation between views in the subspace
        * Is a linear subspace transformation, so not a perfect model for many real-world examples

## 5. Co-training Style Algorithms
### 5.1 Assumptions for Co-training
- Sufficiency, compatibility, conditional independence

**5.1.1 Conditional Independence Assumption**

- Blum and Mitchell, 1998 - Proved co-training can be successful if conditional independence holds.

**5.1.2 Weak Dependence Assumption**

- Abney, 2002 - Found that weak dependence alone can lead to successful co-training.

**5.1.3 Expansion Assumption**

- Balcan et. al., 2004 - With sufficiently strong learning algorithms, even weaker assumptions can be sufficient.

**5.1.3 Large Diversity Assumption**

- Wang and Zhou, 2007 - When learners are more diverse than their errors, performance of classification can be improved by co-training.

### 5.2 Co-training
- Blum and Mitchell, 1998 - Original algorithm.
    * Algorithm
    * L - set of labeled examples
    * U - set of unlabeled examples
    * U' - smaller pool taken from U with u samples
        1. Train 2 naive Bayes classifiers h1 and h2 on the two views in L
        2. Each one then classifies samples in U', and puts its most confident classifications into L
        3. Refill the U' set from U, until it has u samples again
        4. Repeat

### 5.3 Co-EM
- Give unlabeled examples class probabilities
- Has been done with naive Bayes and reformulated SVMs (Brefeld and Sheffer, 2004)

### 5.4 Co-regularization
- Sindhwani, 2005; Sindhwani and Rosenberg, 2008 - Solve optimization problem based on regularizing each predictor as well as enforcing agreement between the predictors.

### 5.5 Co-regression
- Zhou and Li, 2005 - Co-training regression algorithm, CoREG.
    * Uses 2 kNN regressors
- Brefeld et. al. 2006 - Co-regression algorithm

### 5.6 Co-clustering
- Bickel and Scheffer, 2004 - Paper on multi-view version of multiple clustering algorithms.
- Multiple papers on multi-view approaches to spectral clustering

### 5.7 Graph-based Co-training
- Yu et. al. 2007, 2011 - Bayesian undirected graphical model for co-training through Gaussian Process.
- Wang and Zhou, 2010 - Combined the graph-based and disagreement-based frameworks into one, semi-supervised learning model.

### 5.8 Multi-learner Algorithms
- Goldman and Zhou, 2000 - Use a co-training strategy to improve the performance of standard learning methods by using new unlabeled data.
    * Doesn't assume sufficiency
- Zhou and Li, 2005 - Co-training style semi-supervised algorithm called tri-training.
    * Creates 3 classifiers from labeled data, which are refined using unlabeled data based on each other's classifications
- Li et. al., 2006 - Multi-training SVM (MTSVM) designed to mitigate the problem where the SVM's perform poorly when given few labeled feedback samples.
    * Better than just using co-training with SVM, which would require that sub-classifiers have good generalization ability before they can co-train
