import numpy as np
from scipy.stats import ortho_group
import collections
import matplotlib.pyplot as plt

def add_noise(X, n_noise, random_state=None):
    if not random_state is None:
        np.random.seed(random_state)
    noise = np.random.randn(X.shape[0], n_noise)
    return np.hstack((X, noise))

def linear2view(X):
    if X.shape[1] == 1:
        X = -X
    else:
        np.random.seed(2)
        X = X @ ortho_group.rvs(X.shape[1])
    return X

def poly2view(X):
    X = np.asarray([np.power(x, 3) for x in X])
    return X

def polyinv2view(X):
    X = np.asarray([np.cbrt(x) for x in X])
    return X

def sin2view(X):
    X = np.asarray([np.sin(x) for x in X])
    return X


class GaussianMixture:
    def __init__(self, n, mu, sigma, class_probs = None):
        self.mu = mu
        self.sigma = sigma
        self.views = len(mu)
        self.class_probs = class_probs

        if class_probs is None:
            self.latent = np.random.multivariate_normal(mu, sigma, size=n)
            self.y = None
        else:
            self.latent = np.concatenate(
                [
                    np.random.multivariate_normal(
                        self.mu[i], self.sigma[i], size=int(class_probs[i]*n)
                    ) for i in range(len(class_probs))
                ]
            )
            self.y = np.concatenate([i*np.ones(int(class_probs[i]*n))
                                     for i in range(len(class_probs))
                                    ])

    def sample_views(self, n_noise=1, transform='linear', random_states=None):
        """
        Parameters
        ----------
        sn_noise : int, default = 1
            number of noise dimensions to add to transformed latent
        transform : string, default = 'linear'
            Type of transformation to form views
            - value can be 'linear', 'sin', 'polyinv', 'poly', or custom function
        random_states : 1D array-like
            list of seeds to use in generating views, for reproducibility
        """

        if isinstance(transform, collections.Callable):
            X = np.asarray([transform(x) for x in self.latent])
        elif transform == "linear":
            X = linear2view(self.latent)
        elif transform == "poly":
            X = poly2view(self.latent)
        elif transform == "polyinv":
            X = polyinv2view(self.latent)
        elif transform == "sin":
            X = sin2view(self.latent)
        else:
            X = transform(self.latent)

        self.Xs = [X, self.latent]
        for i,X in enumerate(self.Xs):
            if not random_states is None:
                self.Xs[i] = add_noise(X, n_noise=n_noise, 
                    random_state=random_states[i])
            self.Xs[i] = add_noise(X, n_noise=n_noise)

    def plot_2views(self, Xs=None, figsize=(10, 10), title="", show=True):
        if Xs is None:
            Xs = self.Xs
        n = Xs[0].shape[1]
        fig, axes = plt.subplots(n, n, figsize=figsize)
        for i, ax in enumerate(axes.flatten()):
            dim2 = int(i / n)
            dim1 = i % n
            ax.scatter(Xs[0][:, dim1], Xs[1][:, dim2])
            if dim2 == n-1:
                ax.set_xlabel(f"View 1 Dim {dim1+1}")
            if dim1 == 0:
                ax.set_ylabel(f"View 2 Dim {dim2+1}")
            ax.axis('equal')

        plt.suptitle(title)
        if show:
            plt.show()
        else:
            return (fig, axes)

    def plot_latents(self, figsize=(5, 5), title="", show=True):
        n = self.latent.shape[1]
        fig, axes = plt.subplots(n, n, figsize=figsize)
        for i, ax in enumerate(axes.flatten()):
            dim2 = int(i / n)
            dim1 = i % n
            ax.scatter(self.latent[:, dim1], self.latent[:, dim2])
            if dim2 == n-1:
                ax.set_xlabel(f"View 1 Dim {dim1+1}")
            if dim1 == 0:
                ax.set_ylabel(f"View 2 Dim {dim2+1}")

        plt.suptitle(title)
        if show:
            plt.show()
        else:
            return (fig, axes)
