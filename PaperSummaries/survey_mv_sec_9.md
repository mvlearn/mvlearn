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
    * 6 classes (professor, project, etc.)
    * 4 universities
    * 2 views: text on the page, enchor text of hyperlink
  * Citeseer
    * 3312 scientific publication documents
    * 6 classes
    * 3 views: text view, inbound link view, outbound link view
  * UCI Repository
    * 2 classes (advertisements, non advertisements)
    * 6 views: geometry of images, base url, image url, target url, anchor text, alt text
* Empirical Evaluation
  * WebKB 1-6 (different preprocessing steps of the algorithms by different researchers)
    1. (Blum and Mitchell) Co-trained naive Bayes had lower error rate compared to single-view NB
    2. (Nigram and Ghani) Co-EM NB had lower error rate compared to single-view NB and co-trained NB
       * (EM is a generative model, and uses the likelihood-based approach)
    3. (Brefeld and Scheffer) Co-EM based on SVM had lower error rate compared to Co-EM NB, single-view NB, and single-view          SVM
    4. (Sindhwani et al.) Evaluated co-regularization method and compared with single-view regularization method, single-view        SVM and co-trained Laplace SVM. Co-regularization and co-LAP SVM both seemed to do well in terms of mean PRBEP
    5. (Yu et al.) Developed Graphical co-training and Bayesian co-training
    6. (Zhu et al.) Compared multi-view subspace learning vs single-view subspace learning (not huge differences in terms of AUC)
 * UCI (1-4)
   * Several multiple kernel learning methods, such as localized MKL and simple MKL, were evaluated in terms of accuracy and time cost
   * Overall from comparison results, multi-view learning methods improved performance 
 
    
