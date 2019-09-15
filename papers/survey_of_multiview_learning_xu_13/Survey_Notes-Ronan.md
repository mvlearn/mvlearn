# A Survey on Multi-view Learning

### Chang Xu, Dacheng Tao, Chao Xu

https://arxiv.org/abs/1304.5634

## Section 2: Principles for Multi-view Learning

### Consensus Principle

* **Goal**: to maximize the agreement between multiple distinct views
* Given views $X^1$ and $X^2$, the probability of disagreement between two independent hypotheses upper bounds the maximum error of either hypothesis
  * i.e. $P(f^1 \neq f^2) \geq max\{P_{err}(f^1),P_{err}(f^2)\}$
  * Thus can minimize the left to bound the right
* multi-view spectral embedding techniques
  * i.e. kernel canonical correlation analysis (KCCA) 
  * minimize the pairwise distances in low-dimensional embedding space
  * combine w/ SMVs to classify

### Complementary Principle

* **Idea**: each view contains knowledge not captured by other views
* co-training algorithms (semi-supervised)
* Combine different learners on same data
* Multiple kernel learning learns different measures of similarity
* Concatenation of all views -> overfitting on small $n$ and ignores independent statistical properties of views
* Can learn low-dim representation to generate missing data from a view
* Multiview spectral embedding (MSE)

Need to keep in mind both consensus/complementary principles for effective learners

## Section 6: Multiple Kernel Learning (MKL)

### Boosting Methods

* **Kernel**: a function $f: \mathbb{R}^n \times \mathbb{R}^n \rightarrow \mathbb{R}$ 
  * i.e. Matrices ($x^T A x \rightarrow \mathbb{R}$)
* Multiple Additive Regression Kernels
  * Linear combination of kernel functions define
* **Boosting**: Convert weak learners to strong learners by training on residual errors too
* By boosting, can iteratively learn a kernel
  * <https://papers.nips.cc/paper/2202-kernel-design-using-boosting.pdf>

### Semi-Definite Programming (SDP)

* A linear program optimization scheme
* Learn a kernel matrix through SDP

### Quadratically Constrained Quadratic Program (QCQP)

* Support kernel machines (SKM)
* Solves an optimization problem
* Sparsity constraints

### Semi-infinite Linear Program (SILP)

* Different take on QCQP
* Lower computational complexity than SDP, QCQP
* can be solved with off-the-shelf LP solver and standard SVM
* Slow convergence

### Simple MKL

* Sparse and smoothness constraints
* More efficient than SILP
* Hessian MKL even more efficient

### Group-LASSO Approaches

* LASSO effectively a regularization/sparsity constraint on the solution
* Composite Kernel Learning (CKL)
* Takes into account group structure among kernels
* Uses group-LASSO to construct relation

### Bounds for Kernel Learning

* Establishes computational bounds for convex kernels

