from scipy.sparse.linalg import aslinearoperator as scipyaslinearoperator

def convert2scipy(LO):
    """
    returns a scipy linear algebra. 
    """
    return scipyaslinearoperator(LO)