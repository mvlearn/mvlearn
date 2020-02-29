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

# MIT License

# Original work Copyright (c) 2016 Vahid Noroozi
# Modified work Copyright 2019 Zhanghao Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

import warnings
from sklearn.utils import check_X_y, check_array
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler
import time
import logging


def check_Xs(Xs, multiview=False, enforce_views=None):
    """
    Checks Xs and ensures it to be a list of 2D matrices.
    Parameters
    ----------
    Xs : nd-array, list
        Input data.

    multiview : boolean, default (False)
        Throws error if just 1 data matrix

    enforce_views : int, default (not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.

    Returns
    -------
    Xs_converted : object
        The converted and validated X.
    """
    if not isinstance(Xs, list):
        if not isinstance(Xs, np.ndarray):
            msg = "If not list, input must be of type np.ndarray"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    if len(Xs) == 0:
        msg = "Length of input list must be greater than 0"
        raise ValueError(msg)

    if multiview:
        if len(Xs) == 1:
            msg = "Must provide at least two data matrices"
            raise ValueError(msg)
        if enforce_views is not None and len(Xs) != enforce_views:
            msg = "Wrong number of views. Expected {enforce_views} " \
                " but found {len(Xs)}"
            raise ValueError(msg)

    Xs_converted = [check_array(X, allow_nd=False) for X in Xs]

    if not len(set([X.shape[0] for X in Xs_converted])) == 1:
        msg = "All views must have the same number of samples"
        raise ValueError(msg)

    return Xs_converted


def check_Xs_y(Xs, y, multiview=False, enforce_views=None):
    """
    Checks Xs and y for consistent length. Xs is set to be of dimension 3.

    Parameters
    ----------
    Xs : np-array, list
        Input data. F

    y : nd-array, list
        Labels.

    multiview : boolean, default (False)
        Throws error if just 1 data matrix.

    enforce_views : int, default (not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.

    Returns
    -------
    Xs_converted : object
        The converted and validated X.

    y_converted : object
        The converted and validated y.
    """
    Xs_converted = check_Xs(Xs, multiview=multiview,
                            enforce_views=enforce_views)
    _, y_converted = check_X_y(Xs_converted[0], y, allow_nd=False)

    return Xs_converted, y_converted


def check_Xs_y_nan_allowed(
        Xs,
        y,
        multiview=False,
        enforce_views=None,
        num_classes=None
        ):
    """
    Checks Xs and y for consistent length. Xs is set to be of dimension 3.

    Parameters
    ----------
    Xs : nd-array, list
        Input data.
    y : nd-array, list
        Labels.
    multiview : boolean, default (False)
        Throws error if just 1 data matrix
    enforce_views : int, default (not checked)
        If provided, ensures this number of views in Xs. Otherwise not
        checked.
    num_classes: int, default (None)
        Number of classes that must appear in the labels. If none, then
        not checked.

    Returns
    -------
    Xs_converted : object
        The converted and validated X.
    y_converted : object
        The converted and validated y.
    """

    Xs_converted = check_Xs(Xs,
                            multiview=multiview,
                            enforce_views=enforce_views)

    y_converted = np.array(y)
    if len(y_converted) != Xs_converted[0].shape[0]:
        raise ValueError("Incompatible label length {len(y_converted)} for "
                         " data with {Xs_converted[0].shape[0]} samples")

    if num_classes is not None:
        # if not exactly correct number of class labels, raise error
        classes = list(set(y[~np.isnan(y)]))
        n_classes = len(classes)
        if n_classes != num_classes:
            raise ValueError("Wrong number of class labels. Expected {},\
             found {}".format(num_classes, n_classes))

    return Xs_converted, y_converted

class linear_cca():
    """
    Implementation of linear CCA to act on the output of the deep networks
    in DCCA.

    Parameters
    ----------

    Attributes
    -------
    w : list (length=2)
        w[i] : nd-array
        List of the two weight matrices for projecting each view.
    m : list (length=2)
        m[i] : nd-array
        List of the means of the data in each view.
    """
    def __init__(self):
        self.w = [None, None]
        self.m = [None, None]

    def fit(self, H1, H2, outdim_size):
        """
        An implementation of linear CCA.

        Parameters
        ----------
        H1: nd-array, shape (n_samples, n_features)
            View 1 data.
        H2: nd-array, shape (n_samples, n_features)
            View 2 data.
        outdim_size : int (positive)
            The output dimensionality of the CCA transformation.
        """
        r1 = 1e-4
        r2 = 1e-4

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        self.m[0] = np.mean(H1, axis=0)
        self.m[1] = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(self.m[0], (m, 1))
        H2bar = H2 - np.tile(self.m[1], (m, 1))

        SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
        SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T,
                                                 H1bar) + r1 * np.identity(o1)
        SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T,
                                                 H2bar) + r2 * np.identity(o2)

        [D1, V1] = np.linalg.eigh(SigmaHat11)
        [D2, V2] = np.linalg.eigh(SigmaHat22)
        SigmaHat11RootInv = np.dot(
            np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
        SigmaHat22RootInv = np.dot(
            np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

        Tval = np.dot(np.dot(SigmaHat11RootInv,
                                   SigmaHat12), SigmaHat22RootInv)

        [U, D, V] = np.linalg.svd(Tval)
        V = V.T
        self.w[0] = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
        self.w[1] = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
        D = D[0:outdim_size]

    def _get_result(self, x, idx):
        """
        Transform a single view of data based on already fit matrix.

        Parameters
        ----------
        x : nd-array, shape (n_samples, n_features)
            View idx data.
        idx : int
            0 if view 1. 1 if view 2.

        Returns
        -------
        result : nd-array
            Result of linear transformation on input data.
        """
        result = x - self.m[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = np.dot(result, self.w[idx])
        return result

    def test(self, H1, H2):
        """
        Transform inputs based on already fit matrices.

        Parameters
        ----------
        H1 : nd-array, shape (n_samples, n_features)
            View 1 data.
        H2 : nd-array, shape (n_samples, n_features)
            View 2 data.

        Returns
        -------
        Results : list, length=2
            Results of linear transformation on input data.
        """
        return [self._get_result(H1, 0), self._get_result(H2, 1)]


class cca_loss():
    """
    An implementation of the loss function of linear CCA as introduced
    in the original paper for ''DCCA'' [#1Utils].

    Parameters
    ----------
    outdim_size : int (positive)
        The output dimensionality of the CCA transformation.
    use_all_singular_values : boolean
        Whether or not to use all the singular values in the loss calculation.
        If False, only use the top outdim_size singular values.
    device : torch.device object
        The torch device being used in DCCA.

    Attributes
    ----------
    outdim_size : int (positive)
        The output dimensionality of the CCA transformation.
    use_all_singular_values : boolean
        Whether or not to use all the singular values in the loss calculation.
        If False, only use the top outdim_size singular values.
    device : torch.device object
        The torch device being used in DCCA.

    References
    ----------
    .. [#1Utils] Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013, February).
                 Deep canonical correlation analysis. In International conference on
                 machine learning (pp. 1247-1255).
    """
    def __init__(self, outdim_size, use_all_singular_values, device):
        self.outdim_size = outdim_size
        self.use_all_singular_values = use_all_singular_values
        self.device = device

    def loss(self, H1, H2):
        """
        Compute the loss.

        Parameters
        ----------
        H1: nd-array, shape (n_samples, n_features)
                View 1 data.
        H2: nd-array, shape (n_samples, n_features)
            View 2 data.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.t(), H2.t()

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + \
            r1 * torch.eye(o1, device=self.device)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + \
            r2 * torch.eye(o2, device=self.device)

        # Calculating the root inverse of covariance matrices by using
        # eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
            tmp = torch.trace(torch.matmul(Tval.t(), Tval))
            # print(tmp)
            corr = torch.sqrt(tmp)
            # assert torch.isnan(corr).item() == 0
        else:
            # just the top self.outdim_size singular values are used
            U, V = torch.symeig(torch.matmul(
                Tval.t(), Tval), eigenvectors=True)
            # U = U[torch.gt(U, eps).nonzero()[:, 0]]
            U = U.topk(self.outdim_size)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class MlpNet(nn.Module):
    """
    Multilayer perceptron implementation for fully connected network. Used
    by ''DCCA'' for the fully transformation of a single view before linear
    CCA.

    Parameters
    ----------
    layer_sizes : list of ints
        The sizes of the layers of the deep network applied to view 1 before
        CCA. For example, if the input dimensionality is 256, and there is one
        hidden layer with 1024 units and the output dimensionality is 100
        before applying CCA, layer_sizes1=[1024, 100].

    input_size : int (positive)
        The dimensionality of the input vectors to the deep network.

    Attributes
    ----------
    layers : torch.nn.ModuleList object
        The layers in the network.

    """
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                )
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                ))
        print("making MLP in utils.py")
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Feed input forward through layers.

        Parameters
        ----------
        x : torch.tensor
            Input tensor to transform forward.

        Returns
        -------
        x : torch.tensor
            The output after being fed forward through network.
        """
        for layer in self.layers:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    """
    A pair of deep networks for operating on the two views of data. Consists
    of two ''MlpNet'' objects for transforming 2 views of data in ''DCCA''.

    Parameters
    ----------
    layer_sizes1 : list of ints
        The sizes of the layers of the deep network applied to view 1 before
        CCA. For example, if the input dimensionality is 256, and there is one
        hidden layer with 1024 units and the output dimensionality is 100
        before applying CCA, layer_sizes1=[1024, 100].
    layer_sizes2 : list of ints
        The sizes of the layers of the deep network applied to view 2 before
        CCA. Does not need to have the same hidden layer architecture as
        layer_sizes1, but the final dimensionality must be the same.
    input_size1 : int (positive)
        The dimensionality of the input vectors in view 1.
    input_size2 : int (positive)
        The dimensionality of the input vectors in view 2.
    outdim_size : int (positive), default=2
        The output dimensionality of the correlated projections. The deep
        network wil transform the data to this size. If not specified, will
        be set to 2.
    use_all_singular_values : boolean (default=False)
        Whether or not to use all the singular values in the CCA computation
        to calculate the loss. If False, only the top outdim_size singular
        values are used.
    device : string, default='cpu'
        The torch device for processing.

    Attributes
    ----------
    """
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2,
                 outdim_size, use_all_singular_values,
                 device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1 = MlpNet(layer_sizes1, input_size1).double()
        self.model2 = MlpNet(layer_sizes2, input_size2).double()

        self.loss = cca_loss(outdim_size,
                             use_all_singular_values, device).loss

    def forward(self, x1, x2):
        """
        Feed two views of data forward through the respective network.

        Parameters
        ----------
        x1 : torch.tensor, shape=(batch_size, n_features)
            View 1 data to transform.
        x2 : torch.tensor, shape=(batch_size, n_features)
            View 2 data to transform.

        Returns
        -------
        outputs : list, length=2
            - outputs[i] : torch.tensor
            List of the outputs from each view transformation.

        """
        # feature * batch_size
        output1 = self.model1(x1)
        output2 = self.model2(x2)

        return output1, output2
