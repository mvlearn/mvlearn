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
from sklearn.model_selection import train_test_split


def mv_data_split(Xs, test_size=None, train_size=None,
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
    '''

    splits = train_test_split(*Xs, test_size=test_size,
                              train_size=train_size, random_state=random_state,
                              shuffle=shuffle, stratify=stratify)

    Xs_train = list()
    Xs_test = list()
    for i in range(len(splits)):
        if i % 2 == 0:
            Xs_train.append(splits[i])
        else:
            Xs_test.append(splits[i])

    return Xs_train, Xs_test
