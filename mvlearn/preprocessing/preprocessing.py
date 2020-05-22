# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from sklearn import model_selection
from itertools import chain
from ..utils.utils import check_Xs_y, check_Xs


def train_test_split(Xs, y=None, test_size=None, train_size=None,
                     random_state=None, shuffle=True, stratify=None):
    r'''
    Splits multi-view data into random train and test subsets. This
    utility wraps the train_test_split function from
    sklearn.model_selection for ease of use.

    Parameters
    ----------
    Xs : numpy.ndarray or list of array-likes with same shape[0]
        The multiple arrays or views of data to split.

    test_size : float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is set to the complement of the train size. If
        train_size is also None, it will be set to 0.25.

    train_size: float or int, default=None
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test
        size.

    random_state: int or RandomState instance, default=None
        Controls the shuffling applied to the data before applying the
        split. Pass an int for reproducible output across multiple
        function calls.

    shuffle: bool, default=True
        Whether or not to shuffle the data before splitting. If
        shuffle=False then stratify must be None.

    stratify: array-like, default=None
        If not None, data is split in a stratified fashion, using
        this as the class labels.

    Returns
    -------
    Xs_train : list of array-likes
        The subset of the data for training.

    Xs_test : list of array-likes
        The subset of the data for testing.

    y_train : numpy.ndarray or None
        The subset of the targets for training if y was provided.
        Otherwise, None is returned.

    y_test : numpy.ndarray or None
        The subset of the targets for testing if y was provided.
        Otherwise, None is returned.

    Examples
    --------
    >>> from mvlearn.preprocessing import train_test_split
    >>> import numpy as np
    >>> RANDOM_STATE=10
    >>> np.random.seed(RANDOM_STATE)
    >>> data = np.arange(18).reshape((3, 3, 2))
    >>> # Print the data
    >>> for i in range(len(data)):
    ...     print('View %d' % i, data[i], sep='\n')
    View 0
    [[0 1]
    [2 3]
    [4 5]]
    View 1
    [[ 6  7]
    [ 8  9]
    [10 11]]
    View 2
    [[12 13]
    [14 15]
    [16 17]]
    >>> train_Xs, test_Xs, _, _ = train_test_split(data, test_size=0.33,
    ...                                            random_state=RANDOM_STATE)
    >>> # Printing train set
    >>> for i in range(len(data)):
    ...     print('View %d' % i, train_Xs[i], sep='\n')
    View 0
    [[4 5]
    [2 3]]
    View 1
    [[10 11]
    [ 8  9]]
    View 2
    [[16 17]
    [14 15]]
    # Printing test set
    >>> for i in range(len(data)):
    ...     print('View %d' % i, test_Xs[i], sep='\n')

    View 0
    [[0 1]]
    View 1
    [[6 7]]
    View 2
    [[12 1]]
    '''

    if y is None:
        Xs = check_Xs(Xs)
        splits = model_selection.train_test_split(*Xs, test_size=test_size,
                                                  train_size=train_size,
                                                  random_state=random_state,
                                                  shuffle=shuffle,
                                                  stratify=stratify)
    else:
        Xs, y = check_Xs_y(Xs, y)
        splits = model_selection.train_test_split(*Xs, y, test_size=test_size,
                                                  train_size=train_size,
                                                  random_state=random_state,
                                                  shuffle=shuffle,
                                                  stratify=stratify)

    Xs_train = list()
    Xs_test = list()
    for i in range(len(splits)):
        if i % 2 == 0:
            Xs_train.append(splits[i])
        else:
            Xs_test.append(splits[i])

    y_train = y_test = None

    if y is not None:
        y_train = Xs_train.pop()
        y_test = Xs_test.pop()

    return Xs_train, Xs_test, y_train, y_test
