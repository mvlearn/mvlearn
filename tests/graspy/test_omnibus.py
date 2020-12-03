"""
test_py
====================================
"""

import pytest
from mvlearn.embed import Omnibus
import numpy as np

def test_default_constructor():
    Omnibus()

def test_invalid_components():
    with pytest.raises(ValueError):
        Omnibus(n_components=-1)

    with pytest.raises(ValueError):
        Omnibus(n_components=0)

def test_invalid_metric():
    Omnibus(distance_metric="yule")
    with pytest.raises(ValueError):
        Omnibus(distance_metric="blah")

def test_invalid_normalize():
    Omnibus(normalize="l2")
    Omnibus(normalize=None)
    Omnibus(normalize="max")

    with pytest.raises(ValueError):
        Omnibus(normalize="blah")

def test_invalid_algorithm():
    Omnibus(algorithm="full")

    with pytest.raises(ValueError):
        Omnibus(algorithm="blah")

def test_invalid_n_iter():
    with pytest.raises(ValueError):
        Omnibus(n_iter=-1)

    with pytest.raises(ValueError):
        Omnibus(n_iter=0)

def test_embeddings_default_none():
    omni = Omnibus()
    assert omni.embeddings_ == None

def test_omnibus_embedding():
    n_components = 2
    embedder = Omnibus(n_components=n_components)
    n_views = 4
    n = 25
    m = 25
    Xs = []
    for _ in range(n_views):
        X = np.random.rand(n, m)
        Xs.append(X)
    embeddings = embedder.fit_transform(Xs)

    assert len(embeddings) == n_views

def test_omnibus_embedding_no_normalize():
    n_components = 2
    embedder = Omnibus(n_components=n_components, normalize=None)
    n_views = 4
    n = 25
    m = 25
    Xs = []
    for _ in range(n_views):
        X = np.random.rand(n, m)
        Xs.append(X)
    embeddings = embedder.fit_transform(Xs)

    assert len(embeddings) == n_views

def test_fit():
    n_components = 2
    embedder = Omnibus(n_components=n_components)
    n_views = 4
    n = 25
    m = 25
    Xs = []
    for _ in range(n_views):
        X = np.random.rand(n, m)
        Xs.append(X)
    embedder.fit(Xs)
    assert len(embedder.embeddings_ == n_views)