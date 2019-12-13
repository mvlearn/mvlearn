import numpy as np


t = np.random.uniform(-np.pi, np.pi, 80)
e1 = np.random.normal(0, 0.5, (80,2))
e2 = np.random.normal(0, 0.5, (80,2))

x = np.zeros((80,2))
x[:,0] = t
x[:,1] = np.sin(3*t)
x += e1

y = np.zeros((80,2))
y[:,0] = np.exp(t/4)*np.cos(2*t)
y[:,1] = np.exp(t/4)*np.sin(2*t)
y += e2

nSamples=40
train1 = x[:nSamples//2]
train2 = y[:nSamples//2]
test1 = x[nSamples//2:]
test2 = y[nSamples//2:]

d_array = [train1,train2]
d = d_array[0]

def _demean(d):
    """
    Calculates difference from mean of the data

    Parameters
    ----------
    d
        Data of interest (Array)

    Returns
    -------
    diff
        Difference from the mean (Array)
    """
    diff = d - d.mean(0)
    return diff

d = np.nan_to_num(d)
cd = _demean(d)
if ktype == "linear":
    kernel = np.dot(cd, cd.T)
    
elif ktype == "gaussian":
    from scipy.spatial.distance import pdist, squareform

    pairwise_dists = squareform(pdist(cd, "euclidean"))
    kernel = np.exp(-pairwise_dists ** 2 / (2 * sigma ** 2))
elif ktype == "poly":
    kernel = np.dot(cd, cd.T) ** degree
kernel = (kernel + kernel.T) / 2.0
if normalize:
    kernel = kernel / np.linalg.eigvalsh(kernel).max()

