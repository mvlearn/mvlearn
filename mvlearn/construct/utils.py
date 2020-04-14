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


def check_n_views(n_views):
    """
    Checks to make sure n_views is a valid parameter.
    The following checks are made:
    * n_views must be greater than 1.
    * n_views must be an integer.
    The function raises an Exception if a check fails.

    Parameters
    ----------
    n_views: int
        Number of views to construct.
    """

    if n_views < 1:
        raise Exception("n_views must be >= 1.")

    if not isinstance(n_views, int):
        raise Exception("n_views must be an integer.")

    return


def check_n_features(n_features, cols):
    """
    Checks to make sure n_features is a valid parameter.
    The following checks are made:
    * n_features must be postive.
    * n_features must be integer if > 1.
    * n_features must be less than number of columns in data.
    The function raises an Exception if a check fails.

    Parameters
    ----------
    n_features: float, int
        Number of features to randomly select.

    cols: int
        Number of features in data matrix.
    """

    if n_features is None:
        raise Exception("n_features must be specified.")

    if n_features <= 0:
        raise Exception("n_features cannot be 0 or a negative number.")

    if n_features > 1 and not isinstance(n_features, int):
        raise Exception("n_features must be integer if > 1.")

    if n_features > cols:
        raise Exception("n_features must be less than columns in X.")

    return
