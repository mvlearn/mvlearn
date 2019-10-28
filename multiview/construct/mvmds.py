#!/usr/bin/env python
# coding: utf-8

# In[1]:
"""

mvmds.py

====================================


Classical Multiview Multidimensional Scaling

"""



import numpy as np
from sklearn.metrics import euclidean_distances



# In[24]:


def classical_MVMDS(Views,k):
  
    
    """
    Classical Multiview Multidimensional Scaling for jointly reducing 
    the dimensions of multiple views of data. A euclidean distance matrix
    is created for each view, double centered, and the k largest common 
    eigenvectors are returned based on the algorithm proposed by the 
    following paper:
    
    https://www.sciencedirect.com/science/article/pii/S016794731000112X   
    
    
    Parameters
    ----------
    Views : list of matrices, each with the number of rows, n
        The input views
        
    k: Number of dimensions to return
    
    Returns
    -------
    views : list of array-like matrices
        List of constructed views (each matrix has shape [n_rows, n_cols]).
    
    components: A k-dimensional projection of shape [n,k]
    
    """
     
    
    mat = np.ones(shape = (len(Views),len(Views[0]),len(Views[0])))
    
    
    for i in range(len(Views)):
        
        
        view = euclidean_distances(Views[i])  
        view_squared = np.power(np.array(view),2)
        
        J = np.eye(len(view)) - (1/len(view))*np.ones(view.shape) #Centering matrix
        B = -(1/2) * np.matmul(np.matmul(J,view_squared),J)   #Double centered matrix B 
        mat[i] = B
    

    
    components = cpc(k,mat)

    
    return components

def cpc(k,x):
    
    """
    Finds Stepwise Estimation of Common Principal Components as described by
    common Trendafilov implementations based on the following paper:
    
    https://www.sciencedirect.com/science/article/pii/S016794731000112X   
    
    
    Parameters
    ----------
    
    k: Number of dimensions to return
    
    x: List of matrices, each with number of rows, n
    
    
    Returns
    -------
    
    Components: Desired number of Common Principal Components
    
    """
   
    
    
    n = p = x.shape[1]
    
    views = len(x)

    n_num = np.array([n] * views)/np.sum(np.array([n] * views))

    Components = np.zeros((p,k)) 
    
    pi = np.eye(p)

    s = np.zeros((p,p))
    it = 15
    

    #make a for loop for script
    for i in np.arange(views):
        s = s + (n_num[i] * x[i])
        
        
    e1,e2 = np.linalg.eigh(s)
    
    
    q0 = e2[:,::-1] 
    
    
    for i in np.arange(k):
        
        q = q0[:,i]
        q = np.array(q).reshape(len(q),1)
        d = np.zeros((1,views))
    
        for j in np.arange(views):
        
            d[:,j] = np.dot(np.dot(q.T,x[j]),q)
     
    
        for j in np.arange(it):
            s2 = np.zeros((p,p))
        
            for yy in np.arange(views):
                s2 = s2 + (n_num[yy]*np.sum(np.array([n] * views)) * x[yy] / d[:,yy])
         
        
            w = np.dot(s2,q) 
                
            w = np.dot(pi,w)
        
            q = w/ np.sqrt(np.dot(w.T,w))
        
            for yy in np.arange(views):
                
                d[:,yy] = np.dot(np.dot(q.T,x[yy]),q)
            
        Components[:,i] = q[:,0]
        pi = pi - np.dot(q,q.T)


    return(Components)

