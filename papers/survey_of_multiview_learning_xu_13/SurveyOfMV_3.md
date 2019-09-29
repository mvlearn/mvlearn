# A Survey on Multi-View Learning

### Chang Xu, Dacheng Tao, Chao Xu

URL: https://arxiv.org/abs/1304.5634

### Chapters 1 and 2

* In most problems, data is diverse and heterogeneous, but can generally be partitioned into groups
* Conventional machine learning algorithms concatenate multiple views into single view, leading to overfitting and losing structural information of data
* Multiview learning has a function model each view and we jointly optimize each function
* 3-types of multiview learning: co-training, multiple kernel learning, subspace learning
* Co-training
  * Process involves training alternately to maximize mutual agreement on two distinct views of unlabeled data
  * Co-training relies on 3  assumptions: 
    * Sufficiency - each view is sufficient for classification by itself
    * Compatibility - the target function of both views predict the same labels for co-occurring features with high probability
    * Conditional independence - views are conditionally independent given label
* Multiple kernel learning
  * MKL kernels naturally correspond to different views and combining kernels improves ML
* Subspace learning
  * These approaches aim to obtain a latent subspace shared by multiple views by assuming that input views are generated from this latent subspace
    * Dimensionality of subspace lower than any other view -> mitigates curse of dimensionality
  * Latent subspace valuable for inferring views other than observation views
  * Active learning - aims to minimize the amount of labeled data required for learning
  * Domain adaptation - refers to problem of adapting a prediction model trained on one domain to a different target domain
* Principles for Multi-view learning
  * Consensus Principle
    * Aims to maximize agreement between multiple views
    * Probability of disagreement between two independent hypotheses is upper bound of error rate of either hypothesis -> need to minimize rate of disagreement
    * Co-training essentially is based on consensus principle as it minimizes error on labeled examples and maximizes agreement on unlabeled ones
  * Complementary principle:
    * States that in multiview setting, each view of data may contain some knowledge that other views do not have -> multiple view can more comprehensively describe the data
    * Co-training style algorithms can succeed when there are no redundant views
    * When diversity of learners exceeds the amount of errors, co-training style can improve performance of learners
    * As co-training process proceeds, two classifiers become increasingly similar as each one learns from the other
  * Traditional multiview methods
    * Traditionally involved concatenating multiple views into single view
    * Long feature vectors with multiple views can be tackled by constructing latent subspace shared by multiple views

### Chapter 3 

###### View Generation

* Priority for multi-view learning = acquisition of redundant views
* In many cases, multiple views are not naturally available due to certain limitations -> need to create multiple views to support multi-view learning.
* We generate different views by partitioning features into multiple disjoint subsets
  * This is not a trivial problem since simple methods like random selection of features does not always suffice
  * Random subspace method incorporate benefits of bootstrapping and aggregation
  * Several strategies have been proposed to produce multiple views: clustering, random selection, and uniform band slicing
* Pseudo multi-view co-training automatically divides features of a single view dataset into mutually exclusive subsets by enforcing the constraint that at most one of the two corresponding elements in the weight vectors for the two classifiers can be nonzero
* Genetic algorithms can be used by mutating vectors of bits, where a 1 represents that the feature is selected, and a 0 represents that it is not
* 3 main types of view construction methods
  * Construct views through random approaches
  * Reshape or decompose original single-view features into multiple views
  * Methods that perform feature partitioning automatically
* In multi-view feature selection, relationships between multiple views should automatically be considered

##### View Evaluation
* Need to evaluate effectiveness of views for multi-view learning and make sure basic assumptions have not been violated
* Assumption of view sufficiency does not generally hold in practice
  * Main concern of lack of self sufficiency in co-training is that applying additional training data associated with classification noise may corrupt initial classifiers that are doing well
  * Additional classifiers trained on automatically labeled data as well as weighting classifiers can help performance
* To filter and detect view disagreement, we can use conditional view entropy, which will be larger for background noise rather than the foreground -> we can threshold on this when selecting samples
* In multiple kernel learning, different kernels essentially provide different views of the data
  * Combining kernels appears to be a straightforward way of combining disparate information sources; however, these information sources can be noisy, so kernel weights need to be optimized during learning
  * Local techniques rather than global techniques are more useful for dealing with complex noise processes
* Correlation between views is an important consideration in subspace-based approaches
  * CCA describes linear relationship between to views by computing low-dimensional shared embedding that maximizes correlation between variables of the two views
  * CCA can be used to test stochastic independence
  * KCCA is a nonlinear variant of this
