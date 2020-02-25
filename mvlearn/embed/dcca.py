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

from .base import BaseEmbed
from ..utils.utils import check_Xs
from .dcca_utils import *


class DCCA(BaseEmbed):
    """
    An implementation of Deep Canonical Correlation Analysis with PyTorch.
    Computes projections into a common subspace in order to maximize the
    correlation between pairwise projections into the subspace from two views
    of data. Deep CCA can be thought of as using deep networks to learn the
    best potentially nonlinear kernels for a variant of kernel CCA. This
    implementation uses PyTorch.

    Parameters
    ----------
    outdim_size : int (positive), default=2
        The output dimensionality of the correlated projections. The deep
        network wil transform the data to this size. If not specified, will
        be set to 2.
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
    use_all_singular_values : boolean (default=False)
        Whether or not to use all the singular values in the CCA computation
        to calculate the loss. If False, only the top outdim_size singular
        values are used.
    device : string, default='cpu'
        The torch device for processing.
    epoch_num : int (positive)
        The max number of epochs to train the deep networks.
    batch_size : int (positive)
        Batch size for training the deep networks.
    learning_rate : float (positive), default=1e-3
        Learning rate for training the deep networks
    reg_par : float (positive), default=1e-5
        Regularization parameter for training the deep networks
    print_log_info : boolean, default=False
        Whether or not to print the logging info (training loss, epoch time)
        when calling DCCA.fit().

    Attributes
    ----------
    projection_mats_ : list of arrays
        A projection matrix for each view, from the given space to the
        latent space
    ranks_ : list of ints
        number of left singular vectors kept for each view during the first
        SVD

    References
    ----------
    [1] Andrew, G., Arora, R., Bilmes, J., & Livescu, K. (2013, February).
    Deep canonical correlation analysis. In International conference on
    machine learning (pp. 1247-1255).
    """

    def __init__(
            self, outdim_size=2, layer_sizes1=None, layer_sizes2=None,
            input_size1=None, input_size2=None, use_all_singular_values=False,
            device=torch.device('cpu'), epoch_num=10, batch_size=800,
            learning_rate=1e-3, reg_par=1e-5, print_log_info=False
            ):

        super().__init__()
        self.deep_model = DeepCCA(layer_sizes1, layer_sizes2, input_size1,
                                  input_size2, outdim_size,
                                  use_all_singular_values, device=device)

        self.linear_cca = linear_cca()

        self.model = nn.DataParallel(self.deep_model)
        self.model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.loss = self.deep_model.loss
        self.optimizer = torch.optim.RMSprop(self.model.parameters(),
                                             lr=learning_rate,
                                             weight_decay=reg_par)
        self.device = device
        self.outdim_size = outdim_size

        # set up logger for logging training of deep models
        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG,
            format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")
        fh = logging.FileHandler("DCCA.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.print_log_info = print_log_info

    def fit(self, Xs, y=None):
        """
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
        x1.to(self.device)
        x2.to(self.device)

        data_size = x1.size(0)

        checkpoint = 'checkpoint.model'

        train_losses = []
        for epoch in range(self.epoch_num):
            epoch_start_time = time.time()
            self.model.train()
            batch_idxs = list(BatchSampler(RandomSampler(range(data_size)),
                                           batch_size=self.batch_size,
                                           drop_last=False))
            for batch_idx in batch_idxs:
                self.optimizer.zero_grad()
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                loss = self.loss(o1, o2)
                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            train_loss = np.mean(train_losses)

            info_string = "Epoch {:d}/{:d} - time: {:.2f} -"\
                          " training_loss: {:.4f}"

            torch.save(self.model.state_dict(), checkpoint)
            epoch_time = time.time() - epoch_start_time
            if self.print_log_info:
                self.logger.info(info_string.format(
                    epoch + 1, self.epoch_num, epoch_time, train_loss))
        # train_linear_cca
        if self.linear_cca is not None:
            print("x1.shape")
            print(x1.shape)
            _, outputs = self._get_outputs(x1, x2)
            print("outputs[0].shape")
            print(outputs[0].shape)
            print("linear cca")
            self.train_linear_cca(outputs[0], outputs[1])

        checkpoint_ = torch.load(checkpoint)
        self.model.load_state_dict(checkpoint_)

        self.is_fit = True
        return self

    def transform(self, Xs):
        """
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
        """

        if not self.is_fit:
            raise RuntimeError("Must call fit function before transform")
        Xs = check_Xs(Xs, multiview=True)
        x1 = torch.DoubleTensor(Xs[0])
        x2 = torch.DoubleTensor(Xs[1])

        with torch.no_grad():
            losses, outputs = self._get_outputs(x1, x2)
            outputs = self.linear_cca.test(outputs[0], outputs[1])
            return np.mean(losses), outputs

    def train_linear_cca(self, x1, x2):
        self.linear_cca.fit(x1, x2, self.outdim_size)

    def _get_outputs(self, x1, x2):
        with torch.no_grad():
            self.model.eval()
            data_size = x1.size(0)
            batch_idxs = list(BatchSampler(SequentialSampler(range(data_size)),
                              batch_size=self.batch_size,
                              drop_last=False))
            print("batch indices")
            print(len(batch_idxs)) # 88
            print(len(batch_idxs[0])) # 800
            losses = []
            outputs1 = []
            outputs2 = []
            for batch_idx in batch_idxs:
                batch_x1 = x1[batch_idx, :]
                batch_x2 = x2[batch_idx, :]
                o1, o2 = self.model(batch_x1, batch_x2)
                print("batch output size")
                print(o1.shape)
                outputs1.append(o1)
                outputs2.append(o2)
                loss = self.loss(o1, o2)
                losses.append(loss.item())
        outputs = [torch.cat(outputs1, dim=0).cpu().numpy(),
                   torch.cat(outputs2, dim=0).cpu().numpy()]
        print("torch cat size")
        print(torch.cat(outputs1, dim=0).cpu().numpy().shape)
        print(torch.cat(outputs1, dim=0).shape)
        print(outputs[0].shape)

        return outputs

    def print_deep_model_info(self):
        self.logger.info(self.model)
        self.logger.info(self.optimizer)
