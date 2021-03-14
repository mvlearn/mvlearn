"""Nutrimouse dataset loader"""

# Authors: Ronan Perry
#
# License: MIT

from os.path import dirname, join
import numpy as np


def load_nutrimouse(return_Xs_y=True):
    r"""
    Load the Nutrimouse dataset [#1paper], a two-view dataset from a nutrition
    study on mice, as available from
    https://CRAN.R-project.org/package=CCA [#2r].

    Parameters
    ----------
    return_Xs_y : bool, default=True
        If ``True``, returns an ``(Xs, y)`` tuple of the multiple views and
        sample labels.

    Returns
    -------
    (Xs, y) : 2-tuple of the multiple views and labels
        Returned if ``return_Xs_y`` is True (default).

    data : dictionary with the following key: value pairs (see Notes for
        details). Returned if ``return_Xs_y`` is False.
        gene : numpy.ndarray, shape (40, 120)
            The gene expressions (1st view).
        lipid : numpy.ndarray, shape (40, 21)
            The fatty acid concentrations (2nd view)
        genotype : numpy.ndarray, shape (40,)
            The genotype label (1st label).
        diet : numpy.ndarray, shape (40,)
            The diet label (2nd label).
        gene_feature_names : list, length 120
            The names of the genes.
        lipid_feature_names : list, length 21
            The names of the fatty acids.
        genotype_names : list, length 2
            The names of the genotype classes.
        diet_names : list, length 5
            The names of the diet classes.

    Notes
    -----
    This data consists of two views from a nutrition study of 40 mice:

    - gene : expressions of 120 potentially relevant genes
    - lipid : concentrations of 21 hepatic fatty acids

    Each mouse has two labels, four mice per pair of labels:

    - genotype (2 classes) : wild-type, PPARalpha -/-
    - diet (5 classes) : REF, COC, SUN, LIN, FISH

    References
    ----------
    .. [#1paper] P. Martin, H. Guillou, F. Lasserre, S. Déjean, A. Lan, J-M.
            Pascussi, M. San Cristobal, P. Legrand, P. Besse, T. Pineau.
            "Novel aspects of PPARalpha-mediated regulation of lipid and
            xenobiotic metabolism revealed through a nutrigenomic study."
            Hepatology, 2007.

    .. [#2r] González I., Déjean S., Martin P.G.P and Baccini, A. (2008) CCA:
            "An R Package to Extend Canonical Correlation Analysis." Journal
            of Statistical Software, 23(12).

    Examples
    --------
    >>> from mvlearn.datasets import load_nutrimouse
    >>> # Load both views and labels
    >>> Xs, y = load_nutrimouse()
    >>> print(len(Xs))
    2
    >>> print([X.shape for X in Xs])
    [(40, 120), (40, 21)]
    >>> print(labels.shape)
    (40, 2)
    """

    module_path = dirname(__file__)
    folder = "nutrimouse"
    Xs_filenames = ["gene", "lipid"]
    y_filenames = ["genotype", "diet"]
    data = {}

    for fname in Xs_filenames:
        csv_file = join(module_path, folder, fname + '.csv')
        X = np.genfromtxt(csv_file, delimiter=',', dtype=str)
        data[fname] = X[1:].astype(float)
        data[f'{fname}_feature_names'] = list(X[0])

    for fname in y_filenames:
        csv_file = join(module_path, folder, fname + '.csv')
        y = np.genfromtxt(csv_file, delimiter=',', dtype=str)[1:]
        class_names, y = np.unique(y, return_inverse=True)
        data[fname] = y
        data[f'{fname}_names'] = class_names

    if return_Xs_y:
        Xs = [data[X_key] for X_key in Xs_filenames]
        y = np.vstack([data[y_key] for y_key in y_filenames]).T
        return (Xs, y)
    else:
        return data
