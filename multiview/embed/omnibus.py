"""
omnibus.py
==========
Omnibus embedding for multiview dimensionality reduction.
"""

from .base import BaseEmbed
from ..utils.utils import check_Xs

import numpy as np
from graspy.embed import OmnibusEmbed
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

class Omnibus(BaseEmbed):

    def __init__(self, n_components=2, normalize=None, distance_metric = None):
        # TODO: add in omnibus embedding attributes
        super().__init__()
        self.embeddings = None
        self.normalize = normalize
        self.distance_metric = distance_metric


    def check_params(self):
        # TODO: check parameters
        pass

    def fit(self, Xs):
        pass


    def fit_transform(self, Xs):
        dissimilarities = []
        for X in Xs:
            normalized_X = normalize(X, norm="l1")
            dissimilarity = pairwise_distances(normalized_X)
            dissimilarities.append(dissimilarity)

        embedder = OmnibusEmbed()
        self.embeddings = embedder.fit_transform(dissimilarities)
        return self.embeddings


    

