# A Survey on Multi-view Learning
### Chang Xu, Dacheng Tao, Chao Xu
https://arxiv.org/pdf/1304.5634.pdf

# Summary


## Section 8- Applications
There has been strong progress in multiview research. Multiview learning has been used across disciplines in a number of different applications.


![co-training](https://user-images.githubusercontent.com/27905822/64492047-88b65700-d23d-11e9-8219-7b797fd0bfa4.png)


We can split up the applications iin multiview learning into those that integrate co-learning, multiple kernel learning, and subspace learning:
* Co-learning
  * The first application of co-learning was in web document classification. The success of this trial, has lead to a a number of different researchers to implement multiple views into their work.
  * There have been many studies that have explored using multiview learning in ***natural language processing.***
  * It was shown given a small set of labeled training data and a large set of unlabeled data, co-training can reduce the difference in error by around 36%. [Read Here](https://www.aclweb.org/anthology/W01-0501)
  * Using co-training along with simple dimensionality reduction, emotion in language can be predicted using a small set of labeled data. [Read Here](https://www.aclweb.org/anthology/W04-2405)
  * There are also instances of research that have shown multiview learning to be effective in image labeling with limited annotations.
* Multiple kernel learning
  * Object classification was done by linearly combining similarity functions between images for improved classification. [Read Here](https://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.pdf)
  * Multiple kernel learning has been doing very well in improving object recognition
  * Algorithms are being used to ***update and improve new kernels***.
* Subspace learning
  * Subspace learning has proven to be important in analyzing relationships between different views of data
  * This has been used in a number of model search algorithms and facial expression recognition algorithms
  * Researchers were able to represent multiple features for integrating a joint subspace by preserving neighborhood information. [Read Here](https://ieeexplore.ieee.org/abstract/document/6199986/)
  * ***Large computational cost*** problems are being solved in this space
  
