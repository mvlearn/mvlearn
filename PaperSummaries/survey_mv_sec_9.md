# Summary: A Survey on Multiview Learning
URL: https://arxiv.org/abs/1304.5634
### 0. Abstract
* Learn from multi-view data using three approaches:
  * Co-training - maximize mutual agreement on two views of data
  * Multiple kernal training - combine kernels
  * Subspace learning - obtain latent subspace shared by multiple views
* Success of multi-view learning mainly relies on consensus principle and complementary principle
### 1. Introduction 
* Conventional algorithms concatenate multiple views into single view, which results in over-fitting since each view has different statiscial properties.  
* New mv algorithms joinly optimize all functions
* Co-training
 * Co-training relies on:
  * sufficiency - each view sufficient to classify independantly
  * compatibility -  the target function of views predict same labels with high probability
  * conditional independance - views are conditionally independant
* Multiple Kernal Learning
* Subspace learning

 
### 2. Principles on Multi-view Learning
* 
### 9. Performance Evaluation
* Datasets
 * WebKB
  * 8282 academic webpages
  * 6 classes 
  * 4 universities
  * 2 views: text on the page, enchor text of hyperlink
 * Citeseer
  * 3312 scientific publication documents
  * 6 classes
  * 3 views: text view, inbound link view, outbound link view
 * UCI Repository
  * 2 classes
  * 
 * 
