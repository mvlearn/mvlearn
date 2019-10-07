from os.path import dirname, join
import numpy as np


def load_UCI_multifeature():
    """
    Load the UCI multiple features dataset, taken from
    https://archive.ics.uci.edu/ml/datasets/Multiple+Features
    This data set consists of 6 views of handwritten digit images, with
    classes 0-9. The 6 views are the following:

    1. 76 Fourier coefficients of the character shapes
    2. 216 profile correlations
    3. 64 Karhunen-Love coefficients
    4. 240 pixel averages of the images from 2x3 windows
    5. 47 Zernike moments
    6. 6 morphological features

    Each class contains 200 labeled examples.

    Parameters
    ----------

    Returns
    -------
    data : list of np.ndarray, each of size (2000,n_features)
        List of length 6 with each element being the data for one of the
        views.

    labels : np.ndarray
        Array of labels for the digit

    References
    ----------
    [1] M. van Breukelen, R.P.W. Duin, D.M.J. Tax, and J.E. den Hartog, 
    Handwritten digit recognition by combined classifiers, Kybernetika, 
    vol. 34, no. 4, 1998, 381-386
    """

    module_path = dirname(__file__)
    folder = "UCImultifeature"
    filenames = ["mfeat_fou.csv","mfeat_fac.csv","mfeat_kar.csv",
    "mfeat_pix.csv","mfeat_zer.csv","mfeat_mor.csv"]
    for filename in filenames:
        with open(join(module_path, folder, filename)) as csv_file:
            datatemp = np.loadtxt(csv_file, dtype=float)
            data.append(datatemp[:,:-1]) # cut off the labels
            labels = datatemp[:,-1]

    return data, labels