# A Survey on Multi-view Learning
### Chang Xu, Dacheng Tao, Chao Xu
https://arxiv.org/pdf/1304.5634.pdf

# Summary

## Introduction
Multi-view data is data from multiple sources or different feature subsets. Three groups of multi-view learning approaches are (1) co-training, (2) multiple kernel learning and (3) subspace learning. Co-training algorithms train alternately to maximize the mutual agreement on two distinct views of the data. Multiple kernel learning algorithms combine kernels that correspond to different views to improve performance. Subspace learning algorithms generate a latent subspaces shared by multiple views by assuming that the input views are generated from this latent subspace. These approaches all exploit either the consensus principle or complementary principle. 

## Section 3- View Combination
Conventional ML algorithms concatenate all multiple views into a single view to adapt to a single-view learning setting. This causes over-fitting and is not meaningful because each view has a specific statistical property. We use advanced methods of combining multiple views to achieve an improvement in learning performance.

To summarize the various approaches that combine multiple views:
•	Co-training style algorithms
o	Train separate learners on distinct views and then optimized to be consistent across views
o	A late combination of multiple views because the views are considered independently when training the base learners.
•	Multiple kernel learning
o	Calculate separate kernels on each view and then are combined with a kernel-based method
o	Intermediate combination of multiple views because kernels (views) are combined just before or during the training of the learner
•	Subspace learning-based approach
o	Aim to obtain an appropriate subspace by assuming input views are generated from a latent view
o	Prior combination of multiple views because they are considered together to exploit the shared subspace.


In co-training, each classifier trains only on the features of a single view. We maximize the agreement on the predictions of two classifiers on the labeled dataset and minimize the disagreement on the predictions of the two classifiers on the unlabeled dataset. This allows the classifiers to learn from each other and reach an optimal solution. The unlabeled set is considered to be the validation set. In each iteration, the learner on one view labels unlabeled data, which is then used to train the other learner. Two optimal classifiers can be obtained by solving the objective problem to measure the agreement on two distinct views. If a validation set is not provided, we train the classifier on each view and validate the combination of views on the same training set (see Kumar and Daume III, 2011).

![co-training](link-to-image)

In multiple-kernel learning, we use a set of kernel functions and allow an algorithm to choose suitable kernels and the kernel combination. Each kernel corresponds to different inputs coming from different representations, combining kernels is a way to integrate multiple views of information. 

![multiple kernel learning](link-to-image)

There are two categories of ways in which the combination of kernels can be made:
•	Linear combination methods
o	Direct summation kernel- gives equal preference to all kernels
♣	Lanckriet et al. (2002, 2004)
o	Weighted summation kernel- versions approach in the way they place restrictions on kernel weights
♣	Gonen and Alpaydin (2008)
•	Nonlinear combination methods
o	Exponentiation and power
♣	Varma and Babu (2009)
o	Polynomial
♣	Cortes et al. (2009)

In subspace-learning, we assume the input views are generated from a latent subspace to obtain the latent subspace. In single-view learning, PCA is the simplest technique to exploit the subspace from single-view data. In multi-view learning, we use Canonical correlation analysis (CCA) to perform subspace learning. CCA outputs one optimal projection on each view by maximizing the correlation between the two views in the subspace. If datasets exhibit non-linearities, we must use the kernel variate of CCA, called KCCA, which first maps each data point to a higher space in which linear CCA operates. CCA and KCCA exploit the subspace in an unsupervised way, so label information is ignored.

![subspace learning](link-to-image)



