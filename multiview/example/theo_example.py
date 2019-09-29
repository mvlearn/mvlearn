"""
theo_example.py
====================================
Theo's example of how to contribute code to multiview.
"""


def kernel_cca(a, b):
    """
    Runs KCCA for dataview a and b and returns the combined latent space c, the sum of a and b.

    Parameters
    ----------
    a
        The first dataview
    b
	The second dataview
    
    Returns
    -------
    integer
        The combined latent space, aka the sum of a and b 
    """
    c = a + b
    return c
