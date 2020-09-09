import numpy as np
from scipy import linalg
from sklearn.decomposition import PCA


from ..utils.utils import check_Xs
from .base import BaseEstimator


class GroupPCA(BaseEstimator):
    def __init__(self, n_components=None, n_individuals=None, whiten=False,
                 individual_output=True, random_state=None):
        self.n_components = n_components
        self.n_individuals = n_individuals
        self.whiten = whiten
        self.individual_output = individual_output
        self.random_state = random_state

    def fit_transform(self, Xs, y=None):
        Xs = check_Xs(Xs)
        n_features = [X.shape[1] for X in Xs]
        self.n_subjects_ = len(Xs)
        self.n_features_ = n_features
        self.n_samples_ = X.shape[0]
        self.individual_projection_ = self.n_individuals is not None
        if self.n_components is None:
            self.n_components_ = min(n_features)
            if self.individual_projection_:
                self.n_components_ = min(self.n_components_,
                                         self.n_individuals)
        else:
            self.n_components_ = self.n_components

        if self.individual_projection_:
            self.individual_components_ = []
            self.individual_explained_variance_ = []
            self.individual_mean_ = []
            for i, X in enumerate(Xs):
                pca = PCA(self.n_individuals, whiten=self.whiten)
                Xs[i] = pca.fit_transform(X)
                self.individual_components_.append(pca.components_)
                self.individual_explained_variance_.append(
                    pca.explained_variance_
                    )
                self.individual_mean_.append(pca.mean_)
        X_stack = np.hstack(Xs)
        pca = PCA(self.n_components_, whiten=self.whiten)
        output = pca.fit_transform(X_stack)
        self.components_ = pca.components_
        self.explained_variance_ = pca.explained_variance_
        self.mean_ = [np.mean(X, axis=0) for X in Xs]
        if self.individual_output:
            return output
        else:
            self.group_components_ = []
            outputs = []
            for X, mean in zip(Xs, self.mean_):
                X = X - mean
                components = np.dot(X, output).T
                self.group_components_.append(components)
                outputs.append(np.dot(X, components.T))
            return outputs

    def fit(self, Xs, y=None):
        self.fit_transform(Xs, y)
        return self

    def transform(self, Xs):
        if self.individual_projection_:
            for i, (X, mean, components_, explained_variance_) in (
                enumerate(zip(Xs, self.individual_mean_,
                              self.individual_components_,
                              self.individual_explained_variance_s))
                                                                   ):
                X = X - mean
                X_transformed = np.dot(X, components_.T)
                if self.whiten:
                    X_transformed /= np.sqrt(explained_variance_)
                Xs[i] = X_transformed
        Xs = [X - mean for X, mean in zip(Xs, self.mean_)]
        if self.individual_output:
            X_stack = np.hstack(Xs)
            X_transformed = np.dot(X_stack, self.components_.T)
            if self.whiten:
                X_transformed /= np.sqrt(self.explained_variance_)
        else:
            X_transformed = []
            for X, group_components_ in zip(X, self.group_components_):
                X_transformed.append(np.dot(X, group_components_.T))
        return X_transformed
