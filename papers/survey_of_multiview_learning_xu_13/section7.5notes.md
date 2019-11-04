# 7.5 Latent Space Models

## Shared Gaussian Process Latent Variable Model

- what’s a gaussian process?
- Y, Z are two views of the data, X is the latent distribution the Gaussian Processes define how X -> Y and X -> Z. 
- there is some sort of normal distribution constraint — to do with gaussian processes — on the functions that take X->Y and X->Z (eqn. 58 - 61)
- what exactly are these constraints? what do the parameters theta_x, theta_y, theta_z look like for real data (e.g. fMRI data -> latent -> EEG data). how is time series data dealt with? 
- gotta use gradient descent or something similar to maximize the likelihood and get a trained model
- once have trained model, can go from a sample in one view to the latent to a predicted corresponding sample in another view

## Shared Kernel Information Embedding

- “explicit bidirectional probabilistic mappings” — does SGPLVM not have such mappings? perhaps they are not bidirectional
- join distribution that “maximizes mutual information between latent distribution and data distribution” — is the Shannon entropy used for this similar to wasserstein distance? “Mutual information” is definitely the term for a metric as opposed to “mutual information” in a more general sense
- shared KIE maximizes information I: I(x,z) + I(x, y). x \in X, again, is the latent. 
- principal modes -- what are those? 
- eqn (65) says an application of this is human pose inference, where image features are converted to weighted latents, and then those latents are converted into poses and averaged. To use this sKIE technique, they must've used pose-image-features and pose-labels. What would the mutual information maximization equations look like specifically -- i.e. does the data need to be paired?

## Factorized Latent Space

- public and private parts of each view -- not sure what this distinction is
- can encourage sparsity -- making sure a view is not using too much of the latent space, like constraining the dimensionality, I think.
- not sure how the dictionaries work or what's happening in the equations. I see a distance minimization and a regularization term. 

## Latent Space Markov Network

- markov networks and random field theory?
- get a joint distribution of p(x,z,h), where h is latents, other two are views
- this is a model that can be optimized with MLE but there's a function that's easier to optimize -- (73)
