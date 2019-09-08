# A Survey on Multi-view Learning

### Chang Xu, Dacheng Tao, Chao Xu

https://arxiv.org/abs/1304.5634

## Section 2 Notes: Principles for Multi-view Learning

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

