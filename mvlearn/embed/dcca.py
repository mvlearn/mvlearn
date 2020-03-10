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

from .base import BaseEmbed
from ..utils.utils import check_Xs


class linear_cca():
    """
    Implementation of linear CCA to act on the output of the deep networks
    in DCCA.

    Attributes
    ----------
    w_ : list (length=2)
        w[i] : nd-array
        List of the two weight matrices for projecting each view.
    m_ : list (length=2)
        m[i] : nd-array
        List of the means of the data in each view.
    """
    def __init__(self):
        self.w_ = [None, None]
        self.m_ = [None, None]

    def fit(self, H1, H2, n_components):
        """
        An implementation of linear CCA.

        Consider two views :math:`X_1` and :math:`X_2`. Canonical Correlation
        Analysis seeks to find vectors :math:`a_1` and :math:`a_2` to maximize
        the correlation :math:`X_1 a_1` and :math:`X_2 a_2`, expanded below.

        .. math::
            \left(\frac{a_1^TC_{12}a_2}
                {\sqrt{a_1^TC_{11}a_1a_2^TC_{22}a_2}}
                \right)
        where :math:`C_{11}`, :math:`C_{22}`, and :math:`C_{12}` are respectively
        the view 1, view 2, and between view covariance matrix estimates.

        Parameters
        ----------
        H1: nd-array, shape (n_samples, n_features)
            View 1 data.
        H2: nd-array, shape (n_samples, n_features)
            View 2 data.
        n_components : int (positive)
            The output dimensionality of the CCA transformation.
        """
        r1 = 1e-4
        r2 = 1e-4

        m = H1.shape[0]
        o1 = H1.shape[1]
        o2 = H2.shape[1]

        self.m_[0] = np.mean(H1, axis=0)
        self.m_[1] = np.mean(H2, axis=0)
        H1bar = H1 - np.tile(self.m_[0], (m, 1))
        H2bar = H2 - np.tile(self.m_[1], (m, 1))

        # Compute covariance matrices
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
        self.w_[0] = np.dot(SigmaHat11RootInv, U[:, 0:n_components])
        self.w_[1] = np.dot(SigmaHat22RootInv, V[:, 0:n_components])
        D = D[0:n_components]

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
        result = x - self.m_[idx].reshape([1, -1]).repeat(len(x), axis=0)
        result = np.dot(result, self.w_[idx])
        return result

    def transform(self, H1, H2):
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
        results : list, length=2
            Results of linear transformation on input data.
        """
        return [self._get_result(H1, 0), self._get_result(H2, 1)]


class cca_loss():
    """
    An implementation of the loss function of linear CCA as introduced
    in the original paper for ``DCCA`` [#1Utils]_. Details of how this loss
    is computed can be found in the paper or in the documentation for
    ``DCCA``.

    Parameters
    ----------
    n_components : int (positive)
        The output dimensionality of the CCA transformation.
    use_all_singular_values : boolean
        Whether or not to use all the singular values in the loss calculation.
        If False, only use the top n_components singular values.
    device : torch.device object
        The torch device being used in DCCA.

    Attributes
    ----------
    n_components_ : int (positive)
        The output dimensionality of the CCA transformation.
    use_all_singular_values_ : boolean
        Whether or not to use all the singular values in the loss calculation.
        If False, only use the top ``n_components`` singular values.
    device_ : torch.device object
        The torch device being used in DCCA.

    References
    ----------
    .. [#1Utils] Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013,
                 February). Deep canonical correlation analysis. In
                 International conference on machine learning (pp. 1247-1255).
    """
    def __init__(self, n_components, use_all_singular_values, device):
        self.n_components_ = n_components
        self.use_all_singular_values_ = use_all_singular_values
        self.device_ = device

    def loss(self, H1, H2):
        """
        Compute the loss (negative correlation) between 2 views. Details can
        be found in [#1Utils]_ or the documentation for ``DCCA``.

        Parameters
        ----------
        H1: torch.tensor, shape (n_samples, n_features)
            View 1 data.
        H2: torch.tensor, shape (n_samples, n_features)
            View 2 data.
        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        # Transpose matrices so each column is a sample
        H1, H2 = H1.t(), H2.t()

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        # Compute covariance matrices and add diagonal so they are
        # positive definite
        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar, H1bar.t()) + \
            r1 * torch.eye(o1, device=self.device_)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar, H2bar.t()) + \
            r2 * torch.eye(o2, device=self.device_)

        # Calculate the root inverse of covariance matrices by using
        # eigen decomposition
        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        # Additional code to increase numerical stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        # Compute sigma hat matrices using the edited covariance matrices
        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        # Compute the T matrix, whose matrix trace norm is the loss
        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        if self.use_all_singular_values_:
            # all singular values are used to calculate the correlation (and
            # thus the loss as well)
            tmp = torch.trace(torch.matmul(Tval.t(), Tval))
            corr = torch.sqrt(tmp)
        else:
            # just the top self.n_components_ singular values are used to
            # compute the loss
            U, V = torch.symeig(torch.matmul(
                Tval.t(), Tval), eigenvectors=True)
            U = U.topk(self.n_components_)[0]
            corr = torch.sum(torch.sqrt(U))
        return -corr


class MlpNet(nn.Module):
    """
    Multilayer perceptron implementation for fully connected network. Used
    by ``DCCA`` for the fully transformation of a single view before linear
    CCA. Extends `torch.nn.Module <https://pytorch.org/docs/stable/nn.html>`_.

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
    layers_ : torch.nn.ModuleList object
        The layers in the network.

    """
    def __init__(self, layer_sizes, input_size):
        super(MlpNet, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    nn.Sigmoid(),
                ))
        self.layers_ = nn.ModuleList(layers)

    def forward(self, x):
        """
        Feed input forward through layers.

        Parameters
        ----------
        x : torch.tensor
            Input tensor to transform by the network.

        Returns
        -------
        x : torch.tensor
            The output after being fed forward through network.
        """
        for layer in self.layers_:
            x = layer(x)
        return x


class DeepCCA(nn.Module):
    """
    A pair of deep networks for operating on the two views of data. Consists
    of two ``MlpNet`` objects for transforming 2 views of data in ``DCCA``.
    Extends `torch.nn.Module <https://pytorch.org/docs/stable/nn.html>`_.

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
    n_components : int (positive), default=2
        The output dimensionality of the correlated projections. The deep
        network will transform the data to this size. If not specified, will
        be set to 2.
    use_all_singular_values : boolean (default=False)
        Whether or not to use all the singular values in the CCA computation
        to calculate the loss. If False, only the top ``n_components`` singular
        values are used.
    device : string, default='cpu'
        The torch device for processing.

    Attributes
    ----------
    model1_ : ``MlpNet`` object
        Deep network for view 1 transformation.
    model2_ : ``MlpNet`` object
        Deep network for view 2 transformation.
    loss_ : ``cca_loss`` object
        Loss function for the 2 view DCCA.
    """
    def __init__(self, layer_sizes1, layer_sizes2, input_size1, input_size2,
                 n_components, use_all_singular_values,
                 device=torch.device('cpu')):
        super(DeepCCA, self).__init__()
        self.model1_ = MlpNet(layer_sizes1, input_size1).double()
        self.model2_ = MlpNet(layer_sizes2, input_size2).double()

        self.loss_ = cca_loss(n_components,
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
        output1 = self.model1_(x1)
        output2 = self.model2_(x2)

        return output1, output2


class DCCA(BaseEmbed):
    r"""
    An implementation of Deep Canonical Correlation Analysis [#1DCCA]_ with
    PyTorch. It computes projections into a common subspace in order to
    maximize the correlation between pairwise projections into the subspace
    from two views of data. To obtain these projections, two fully connected
    deep networks are trained to initially transform the two views of data.
    Then, the transformed data is projected using linear CCA. This can be
    thought of as training a kernel for each view that initially acts on the
    data before projection. The networks are trained to maximize the ability
    of the linear CCA to maximize the correlation between the final
    dimensions.

    Parameters
    ----------
    input_size1 : int (positive)
        The dimensionality of the input vectors in view 1.
    input_size2 : int (positive)
        The dimensionality of the input vectors in view 2.
    n_components : int (positive), default=2
        The output dimensionality of the correlated projections. The deep
        network wil transform the data to this size. Must satisfy:
        ``n_components`` <= max(layer_sizes1[-1], layer_sizes2[-1]).
    layer_sizes1 : list of ints, default=None
        The sizes of the layers of the deep network applied to view 1 before
        CCA. For example, if the input dimensionality is 256, and there is one
        hidden layer with 1024 units and the output dimensionality is 100
        before applying CCA, layer_sizes1=[1024, 100]. If ``None``, set to
        [1000, ``self.n_components_``].
    layer_sizes2 : list of ints, default=None
        The sizes of the layers of the deep network applied to view 2 before
        CCA. Does not need to have the same hidden layer architecture as
        layer_sizes1, but the final dimensionality must be the same. If
        ``None``, set to [1000, ``self.n_components_``].
    use_all_singular_values : boolean (default=False)
        Whether or not to use all the singular values in the CCA computation
        to calculate the loss. If False, only the top n_components singular
        values are used.
    device : string, default='cpu'
        The torch device for processing.
    epoch_num : int (positive)
        The max number of epochs to train the deep networks.
    batch_size : int (positive)
        Batch size for training the deep networks.
    learning_rate : float (positive), default=1e-3
        Learning rate for training the deep networks.
    reg_par : float (positive), default=1e-5
        Weight decay parameter used in the RMSprop optimizer.
    threshold : float, default=1e-2
        Threshold difference between successive iteration losses to define
        convergence and stop training.
    print_train_log_info : boolean, default=False
        Whether or not to print the logging info (training loss at each epoch)
        when calling DCCA.fit().

    Attributes
    ----------
    input_size1_ : int (positive)
        The dimensionality of the input vectors in view 1.
    input_size2_ : int (positive)
        The dimensionality of the input vectors in view 2.
    n_components_ : int (positive), default=2
        The output dimensionality of the correlated projections. The deep
        network wil transform the data to this size. If not specified, will
        be set to 2.
    layer_sizes1_ : list of ints
        The sizes of the layers of the deep network applied to view 1 before
        CCA. For example, if the input dimensionality is 256, and there is one
        hidden layer with 1024 units and the output dimensionality is 100
        before applying CCA, layer_sizes1=[1024, 100].
    layer_sizes2_ : list of ints
        The sizes of the layers of the deep network applied to view 2 before
        CCA. Does not need to have the same hidden layer architecture as
        layer_sizes1, but the final dimensionality must be the same.
    use_all_singular_values_ : boolean (default=False)
        Whether or not to use all the singular values in the CCA computation
        to calculate the loss. If False, only the top n_components singular
        values are used.
    device_ : string, default='cpu'
        The torch device for processing.
    epoch_num_ : int (positive)
        The max number of epochs to train the deep networks.
    batch_size_ : int (positive)
        Batch size for training the deep networks.
    learning_rate_ : float (positive), default=1e-3
        Learning rate for training the deep networks
    reg_par_ : float (positive), default=1e-5
        Weight decay parameter used in the RMSprop optimizer.
    print_train_log_info_ : boolean, default=False
        Whether or not to print the logging info (training loss at each epoch)
        when calling DCCA.fit().
    deep_model_ : ``DeepCCA`` object
        2 view Deep CCA object used to transform 2 views of data together.
    linear_cca_ : ``linear_cca`` object
        Linear CCA object used to project final transformations from output
        of ``deep_model`` to the ``n_components``.
    model_ : torch.nn.DataParallel object
        Wrapper around ``deep_model`` to allow parallelisation.
    loss_ : ``cca_loss`` object
        Loss function for ``deep_model``. Defined as the negative correlation
        between outputs of transformed views.
    optimizer_ : torch.optim.RMSprop object
        Optimizer used to train the networks.
    threshold_ : float, default=1e-3
        Threshold difference between successive iteration losses to define
        convergence and stop training.

    Notes
    -----
    Deep Canonical Correlation Analysis is a method of finding highly
    correlated subspaces for 2 views of data using nonlinear transformations
    learned by deep networks. It can be thought of as using deep networks
    to learn the best potentially nonlinear kernels for a variant of kernel
    CCA.

    The networks used for each view in DCCA consist of fully connected linear
    layers with a sigmoid activation function.

    The problem DCCA problem is formulated from [#1DCCA]_. Consider two
    views :math:`X_1` and :math:`X_2`. DCCA seeks to find the parameters for
    each view, :math:`\Theta_1` and :math:`\Theta_2`, such that they maximize

    .. math::
        \text{corr}\left(f_1\left(X_1;\Theta_1\right),
        f_2\left(X_2;\Theta_2\right)\right)

    These parameters are estimated in the deep network by following gradient
    descent on the input data. Taking :math:`H_1, H_2 \in R^{o \times m}` to
    be the outputs of the deep network in each column for the input data of
    size :math:`m`. Take the centered matrix :math:`\bar{H}_1 =
    H_1-\frac{1}{m}H_1{1}`, and :math:`\bar{H}_2 = H_2-\frac{1}{m}H_2{1}`.
    Then, define

    .. math::
        \begin{align*}
        \hat{\Sigma}_{12} &= \frac{1}{m-1}\bar{H}_1\bar{H}_2^T \\
        \hat{\Sigma}_{11} &= \frac{1}{m-1}\bar{H}_1\bar{H}_1^T + r_1I \\
        \hat{\Sigma}_{22} &= \frac{1}{m-1}\bar{H}_2\bar{H}_2^T + r_2I
        \end{align*}

    Where :math:`r_1` and :math:`r_2` are regularization constants :math:`>0`
    so the matrices are guaranteed to be positive definite.

    The correlation objective function is the sum of the top :math:`k`
    singular values of the matrix :math:`T`, where

    .. math::
        T = \hat{\Sigma}_{11}^{-1/2}\hat{\Sigma}_{12}\hat{\Sigma}_{22}^{-1/2}

    Which is the matrix norm of T. Thus, the loss is

    .. math::
        L(X_1, X2) = -\text{corr}\left(H_1, H_2\right) =
        -\text{tr}(T^TT)^{1/2}.

    Examples
    --------
    >>> import numpy as np
    >>> from mvlearn.embed.dcca import DCCA
    >>> view1 = np.exp(np.random.normal(size=(1000, 100)))
    >>> view2 = np.random.normal(loc=2, size=(1000, 75))
    >>> input_size1, input_size2 = 100, 75
    >>> n_components = 2
    >>> layer_sizes1 = [1024, 2]
    >>> layer_sizes2 = [1024, 2]
    >>> dcca = DCCA(input_size1, input_size2, n_components, layer_sizes1,
                    layer_sizes2)
    >>> outputs = dcca.fit_transform([view1, view2])
    >>> print(outputs[0].shape)
    (1000, 2)

    References
    ----------
    .. [#1DCCA] Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013,
                February). Deep canonical correlation analysis. In
                International conference on machine learning (pp. 1247-1255).
    """

    def __init__(
            self, input_size1=None, input_size2=None, n_components=2,
            layer_sizes1=None, layer_sizes2=None,
            use_all_singular_values=False, device=torch.device('cpu'),
            epoch_num=50, batch_size=800, learning_rate=1e-3, reg_par=1e-5,
            threshold=1e-3, print_train_log_info=False
            ):

        super().__init__()
        # check input_size1/2

        self.input_size1_ = input_size1
        self.input_size2_ = input_size2
        self.n_components_ = n_components

        if layer_sizes1 is None:
            self.layer_sizes1_ = [1000, n_components]
        if layer_sizes2 is None:
            self.layer_sizes2_ = [1000, n_components]

        self.use_all_singular_values_ = use_all_singular_values
        self.device_ = device
        self.epoch_num_ = epoch_num
        self.batch_size_ = batch_size
        self.learning_rate_ = learning_rate
        self.reg_par_ = reg_par
        self.print_train_log_info_ = print_train_log_info
        self.threshold_ = threshold

        self.deep_model_ = DeepCCA(layer_sizes1, layer_sizes2, input_size1,
                                   input_size2, n_components,
                                   use_all_singular_values, device=device)
        self.linear_cca_ = linear_cca()

        self.model_ = nn.DataParallel(self.deep_model_)
        self.model_.to(device)
        self.loss_ = self.deep_model_.loss_
        self.optimizer_ = torch.optim.RMSprop(self.model_.parameters(),
                                              lr=self.learning_rate_,
                                              weight_decay=self.reg_par_)

    def fit(self, Xs, y=None):
        r"""
        Fits the deep networks for each view such that the output of the
        linear CCA has maximum correlation.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            The data to fit to. Each view will receive its own embedding.

        y : Unused parameter for base class fit_transform compliance

        Returns
        -------
        self : returns an instance of self.
        """
        Xs = check_Xs(Xs, multiview=True)  # ensure valid input
        x1 = torch.DoubleTensor(Xs[0])
        x2 = torch.DoubleTensor(Xs[1])
        x1.to(self.device_)
        x2.to(self.device_)

        data_size = x1.size(0)

        checkpoint = 'checkpoint.model'

        train_losses = []
        epoch = 0
        current_loss = np.inf
        train_loss = 1
        while (current_loss - train_loss > self.threshold_)\
                and epoch < self.epoch_num_:
            self.model_.train()
            batch_idxs = list(BatchSampler(RandomSampler(range(data_size)),
                                           batch_size=self.batch_size_,
                                           drop_last=False))
            current_loss = train_loss
            for batch_idx in batch_idxs:
                self.optimizer_.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model_(batch_x1, batch_x2)
                loss = self.loss_(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer_.step()
            train_loss = np.mean(train_losses)
            if self.print_train_log_info_:
                info_string = "Epoch {:d}/{:d},"\
                    " training_loss: {:.4f}"
                print(info_string.format(epoch + 1, self.epoch_num_,
                      train_loss))

            torch.save(self.model_.state_dict(), checkpoint)
            epoch += 1

        # train_linear_cca
        if self.linear_cca_ is not None:
            losses, outputs = self._get_outputs(x1, x2)
            self._train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model_.load_state_dict(checkpoint_)

        self.is_fit = True
        return self

    def transform(self, Xs, return_loss=False):
        r"""
        Embeds data matrix(s) using the trained deep networks and fitted CCA
        projection matrices. May be used for out-of-sample embeddings.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray
             - Xs length: n_views
             - Xs[i] shape: (n_samples, n_features_i)
            A list of data matrices from each view to transform based on the
            prior fit function. If view_idx defined, then Xs is a 2D data
            matrix corresponding to a single view.

        Returns
        -------
        Xs_transformed : list of array-likes or array-like
            Transformed samples. Same structure as Xs, but potentially
            different n_features_i.
        loss : float
            Average loss over data, defined as negative correlation of
            transformed views. Only returned if ``return_loss=True``.
        """

        if not self.is_fit:
            raise RuntimeError("Must call fit function before transform")
        Xs = check_Xs(Xs, multiview=True)
        x1 = torch.DoubleTensor(Xs[0])
        x2 = torch.DoubleTensor(Xs[1])

        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)
            outputs = self.linear_cca_.transform(outputs[0], outputs[1])
            if return_loss:
                return outputs, np.mean(losses)
            return outputs

    def _train_linear_cca(self, x1, x2):
        """
        Private function to fit the linear CCA model for use after the
        deep layers.

        Parameters
        ----------
        x1 : torch.tensor
            Input view 1 data.
        x2 : torch.tensor
            Input view 2 data.
        """
        self.linear_cca_.fit(x1, x2, self.n_components_)

    def _get_outputs(self, x1, x2):
        """
        Private function to get the transformed data and the corresponding
        loss for the given inputs.

        Parameters
        ----------
        x1 : torch.tensor
            Input view 1 data.
        x2 : torch.tensor
            Input view 2 data.

        Returns
        -------
        losses : list
            List of losses for each batch taken from the input data.
        outputs : list of tensors
            outputs[i] is the output of the deep models for view i.
        """
        with torch.no_grad():
            self.model_.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(range(data_size)),
                              batch_size=self.batch_size_,
                              drop_last=False))
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model_(batch_x1, batch_x2)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss_(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]

        return losses, outputs
