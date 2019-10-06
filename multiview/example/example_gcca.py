import sys

sys.path.append("../..")
from multiview.utils.utils import check_Xs
from multiview.embed.gcca import GCCA
import sklearn
from sklearn.decomposition import PCA
import numpy as np

## Generate data with 2 views
np.random.seed(0)
n_obs = 4
n_views = 2
n_features = 6
X = np.random.normal(0, 1, size=(n_views, n_obs, n_features))

## Fit projections to data and project
gcca = GCCA()
gcca = gcca.fit(X)
projs = gcca.transform(X)

## PCA Data
X0 = PCA(n_components=projs.shape[2]).fit_transform(X[0])
X1 = PCA(n_components=projs.shape[2]).fit_transform(X[1])

## Print Distances
print(f"Raw Distance: {np.linalg.norm(X[0] - X[1])}")
print(f"PCA Distance: {np.linalg.norm(X0[0] - X1[1])}")
print(f"GCCA Distance: {np.linalg.norm(projs[0] - projs[1])}")

