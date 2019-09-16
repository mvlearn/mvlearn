#TODO: Copyright/license

__author__ = 'Ronan Perry'


import numpy as np
from scipy import linalg,stats
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize

from tqdm import tqdm


def _preprocess(x):
    """
    Subtracts the row means and divides by the row standard deviations.
    Then subtracts column means.
    
    Parameters
    ----------
    x : array-like, shape (n_regions, n_features)
        The data to preprocess
    """
    
    # Mean along rows using sample mean and sample std
    x2 = stats.zscore(x,axis=1,ddof=1) 
    # Mean along columns
    mu = np.mean(x2,axis=0)
    x2 -= mu
    return(x2)

def gcca(data, percent_var=0.9, rank_tolerance=None, n_components=None, tall=False, return_meta=False, verbose=False):
    """
    An implementation of Generalized Canonical Correalation Analysis.Computes individual 
    projections into a common subspace such that the correlations between pairwise projections 
    are minimized (ie. maximize pairwise correlation). Reduces to CCA in the case of two samples.
    
    See https://www.sciencedirect.com/science/article/pii/S1053811912001644?via%3Dihub
    for relevant details.
    
    Example:
    import numpy as np
    from gcca import gcca
    
    X1 = np.random.normal(0,1,size=(10,100))
    X2 = np.random.normal(0,1,size=(10,200))
    projections = gcca([X1, X2])
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_regions, n_features)
        The data to embed. Each sample will receive its own embedding.
    percent_var : percent, default=0.9
        Explained variance for rank selection during initial SVD of each sample.
    rank_tolerance : float, optional, default=None
        Singular value threshold for rank selection during initial SVD of each sample.
    n_components : int (postivie), optional, default=None
        Rank to truncate to during initial SVD of each sample.
    tall : boolean, default=False
        Set to true if #rows > #columns, speeds up SVD
    return_meta : boolean, default=False
        Whether to return run information at the end or not
    """
    n = data[0].shape[0]

    data = [_preprocess(x) for x in data]
    
    Uall = []
    Sall = []
    Vall = []
    ranks = []
    
    for x in tqdm(data, disable=(not verbose)):
        # Preprocess
        x[np.isnan(x)] = 0

        # compute the SVD of the data
        if tall:
            v,s,ut = linalg.svd(x.T, full_matrices=False)
        else:
            u,s,vt = linalg.svd(x, full_matrices=False)
            ut = u.T; v = vt.T
        
        Sall.append(s)
        Vall.append(v)
        # Dimensions to reduce to
        if rank_tolerance:
            rank = sum(s > rank_tolerance)
        elif n_components:
            rank = n_components
        else:
            s2 = np.square(s)
            rank = sum(np.cumsum(s2/sum(s2)) < percent_var) + 1
        ranks.append(rank)
        
        u = ut.T[:,:rank]
        Uall.append(u)

    d = min(ranks)

    # Create a concatenated view of Us
    Uall_c = np.concatenate(Uall,axis=1)

    _,_,VV=svds(Uall_c,d)
    VV = np.flip(VV.T,axis=1)
    VV = VV[:,:min([d,VV.shape[1]])]

    # SVDS the concatenated Us
    idx_end = 0
    projX = []
    As = []
    for i in range(len(data)):
        idx_start = idx_end
        idx_end = idx_start + ranks[i]
        VVi = normalize(VV[idx_start:idx_end,:],'l2',axis=0)
        
        # Compute the canonical projections
        A = np.sqrt(n-1) * Vall[i][:,:ranks[i]]
        A = A @ (linalg.solve(np.diag(Sall[i][:ranks[i]]), VVi))
        projX.append(data[i] @ A)
        As.append(A)
    
    if return_meta:
        return(projX,ranks,As)
    else:
        return(projX)