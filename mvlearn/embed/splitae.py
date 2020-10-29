# License: MIT

import sys
import itertools

import numpy as np
# XXX  I would use a nested import for matplotlib to make it a soft dep
import matplotlib.pyplot as plt
import tqdm

try:
    import torch
except ModuleNotFoundError as error:
    print(
        f"Error: {error}. torch dependencies required for this function. \
    Please consult the mvlearn installation instructions at \
    https://github.com/mvlearn/mvlearn to correctly install torch \
    dependencies."
    )
    sys.exit(1)

from .base import BaseEmbed
from ..utils.utils import check_Xs


class _FullyConnectedNet(torch.nn.Module):
    r"""
    General torch module for a fully connected neural network.
    - input_size: number of nodes in the first layer
    - num_hidden_layers: number of hidden layers
    - hidden_size: number of nodes in each hidden layer
    - embedding_size: number of nodes in the output layer.
    All are ints. Each hidden layer has the same number of nodes.
    """

    def __init__(
        self, input_size, hidden_size, num_hidden_layers, embedding_size
    ):
        super().__init__()
        assert num_hidden_layers >= 0, "can't have negative hidden layer count"
        assert hidden_size >= 1, "hidden size must involve >= 1 node"
        assert embedding_size >= 1, "embedding size must involve >= 1 node"
        self.layers = torch.nn.ModuleList()
        if num_hidden_layers == 0:
            self.layers.append(torch.nn.Linear(input_size, embedding_size))
        else:
            self.layers.append(torch.nn.Linear(input_size, hidden_size))
            for i in range(num_hidden_layers - 1):
                self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(torch.nn.Linear(hidden_size, embedding_size))

    def forward(self, x):
        # Forward pass for the network. Pytorch automatically calculates
        # backwards pass
        for layer in self.layers[:-1]:
            x = torch.nn.Sigmoid()(layer(x))
        x = self.layers[-1](x)  # no activation on last layer
        return x

    def param_count(self):
        return np.sum([np.prod(s.shape) for s in self.parameters()])


class SplitAE(BaseEmbed):
    r"""
    Implements an autoencoder that creates an embedding of a view View1 and
    from that embedding reconstructs View1 and another view View2, as
    described in [#1Split]_.

    Parameters
    ----------
    hidden_size : int (default=64)
        number of nodes in the hidden layers
    num_hidden_layers : int (default=2)
        number of hidden layers in each encoder or decoder net
    embed_size : int (default=20)
        size of the bottleneck vector in the autoencoder
    training_epochs : int (default=10)
        how many times the network trains on the full dataset
    batch_size : int (default=16):
        batch size while training the network
    learning_rate : float (default=0.001)
        learning rate of the Adam optimizer
    print_info : bool (default=True)
        whether or not to print errors as the network trains.
    print_graph : bool (default=True)
        whether or not to graph training loss

    Attributes
    ----------
    view1_encoder_ : torch.nn.Module
        the View1 embedding network as a PyTorch module
    view1_decoder_ : torch.nn.Module
        the View1 decoding network as a PyTorch module
    view2_decoder_ : torch.nn.Module
        the View2 decoding network as a PyTorch module

    Warns
    -----
    In order to run SplitAE, pytorch and other certain optional dependencies
    must be installed. See the installation page for details.

    Notes
    -----
    .. figure:: /figures/splitAE.png
        :width: 250px
        :alt: SplitAE diagram
        :align: center

    In this figure :math:`\textbf{x}` is View1 and :math:`\textbf{y}`
    is View2

    Each encoder / decoder network is a fully connected neural net with
    paramater count equal to:

    .. math::
        \left(\text{input_size} + \text{embed_size}\right) \cdot
        \text{hidden_size} +
        \sum_{1}^{\text{num_hidden_layers}-1}\text{hidden_size}^2

    Where :math:`\text{input_size}` is the number of features in View1
    or View2.

    The loss that is reduced via gradient descent is:

    .. math::
        J = \left(p(f(\textbf{x})) - \textbf{x}\right)^2 +
        \left(q(f(\textbf{x})) - \textbf{y}\right)^2

    Where :math:`f` is the encoder, :math:`p` and :math:`q` are
    the decoders, :math:`\textbf{x}` is View1,
    and :math:`\textbf{y}` is View2.

    References
    ----------
    .. [#1Split] Weiran Wang, Raman Arora, Karen Livescu, and Jeff Bilmes.
        "`On Deep Multi-View Representation Learning.
        <http://proceedings.mlr.press/v37/wangb15.pdf>`_",
        ICML, 2015.

    For more extensive examples, see the ``tutorials`` for SplitAE in this
    documentation.
    """

    def __init__(
        self,
        hidden_size=64,
        num_hidden_layers=2,
        embed_size=20,
        training_epochs=10,
        batch_size=16,
        learning_rate=0.001,
        print_info=False,
        print_graph=True,
    ):
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_hidden_layers = num_hidden_layers
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.print_info = print_info
        self.print_graph = print_graph

    def fit(self, Xs, validation_Xs=None, y=None):
        r"""
        Given two views, create and train the autoencoder.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray.
             - Xs[0] is View1 and Xs[1] is View2
             - Xs length: n_views, only 2 is currently supported for splitAE.
             - Xs[i] shape: (n_samples, n_features_i)
        validation_Xs : list of array-likes or numpy.ndarray
            optional validation data in the same shape of Xs. If
            :code:`print_info=True`, then validation error, calculated with
            this data, will be printed as the network trains.
        y : ignored
            Included for API compliance.
        """

        Xs = check_Xs(Xs, multiview=True, enforce_views=2)
        assert (
            Xs[0].shape[0] >= self.batch_size
        ), """batch size must be <= to
            number of samples"""
        assert self.batch_size > 0, """can't have negative batch size"""
        assert (
            self.training_epochs >= 0
        ), """can't train for negative amount of
            times"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        view1 = torch.FloatTensor(Xs[0])
        view2 = torch.FloatTensor(Xs[1])

        self.view1_encoder_ = _FullyConnectedNet(
            view1.shape[1], self.hidden_size,
            self.num_hidden_layers, self.embed_size
        ).to(device)
        self.view1_decoder_ = _FullyConnectedNet(
            self.embed_size, self.hidden_size,
            self.num_hidden_layers, view1.shape[1]
        ).to(device)
        self.view2_decoder_ = _FullyConnectedNet(
            self.embed_size, self.hidden_size,
            self.num_hidden_layers, view2.shape[1]
        ).to(device)

        self.view1_encoder_ = self.view1_encoder_
        self.view1_decoder_ = self.view1_decoder_
        self.view2_decoder_ = self.view2_decoder_

        if self.print_graph:
            print(
                "Parameter counts: \nview1_encoder: {:,}\nview1_decoder: {:,}"
                "\nview2_decoder: {:,}".format(
                    self.view1_encoder_.param_count(),
                    self.view1_decoder_.param_count(),
                    self.view2_decoder_.param_count(),
                )
            )

        parameters = [
            self.view1_encoder_.parameters(),
            self.view1_decoder_.parameters(),
            self.view2_decoder_.parameters(),
        ]
        optim = torch.optim.Adam(
            itertools.chain(*parameters), lr=self.learning_rate
        )
        n_samples = view1.shape[0]
        epoch_train_errors = []
        epoch_test_errors = []

        for epoch in tqdm.tqdm(
            range(self.training_epochs), disable=(not self.print_info)
        ):
            batch_errors = []
            for batch_num in range(n_samples // self.batch_size):
                optim.zero_grad()
                view1_batch = view1[
                    batch_num * self.batch_size:
                    (batch_num + 1) * self.batch_size
                ]
                view2_batch = view2[
                    batch_num * self.batch_size:
                    (batch_num + 1) * self.batch_size
                ]
                embedding = self.view1_encoder_(view1_batch.to(device))
                view1_reconstruction = self.view1_decoder_(embedding)
                view2_reconstruction = self.view2_decoder_(embedding)
                view1_error = torch.nn.MSELoss()(
                    view1_reconstruction, view1_batch.to(device)
                )
                view2_error = torch.nn.MSELoss()(
                    view2_reconstruction, view2_batch.to(device)
                )
                total_error = view1_error + view2_error
                total_error.backward()
                optim.step()
                batch_errors.append(total_error.item())
            if self.print_info:
                print(
                    "Average train error during epoch {} was {}".format(
                        epoch, np.mean(batch_errors)
                    )
                )
            epoch_train_errors.append(np.mean(batch_errors))
            if validation_Xs is not None:
                test_error = self._test_error(validation_Xs)
                if self.print_info:
                    print(
                        "Average test  error during epoch {} was {}\n".format(
                            epoch, test_error
                        )
                    )
                epoch_test_errors.append(test_error)

        if self.print_graph:
            plt.plot(epoch_train_errors, label="train error")
            if validation_Xs is not None:
                plt.plot(epoch_test_errors, label="test error")
            plt.title("Errors during training")
            plt.xlabel("Epoch")
            plt.ylabel("Error")
            plt.legend()
            plt.show()
        return self

    def _test_error(self, Xs):
        # Calculates the error of the network on a set of data Xs
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        n_samples = Xs[0].shape[0]
        validation_batch_size = self.batch_size
        test_indices = np.random.choice(
            n_samples, validation_batch_size, replace=False
        )
        view1_batch = torch.FloatTensor(Xs[0][test_indices])
        view2_batch = torch.FloatTensor(Xs[1][test_indices])
        with torch.no_grad():
            embedding = self.view1_encoder_(view1_batch.to(device))
            view1_reconstruction = self.view1_decoder_(embedding)
            view2_reconstruction = self.view2_decoder_(embedding)
            view1_error = torch.nn.MSELoss()(
                view1_reconstruction, view1_batch.to(device)
            )
            view2_error = torch.nn.MSELoss()(
                view2_reconstruction, view2_batch.to(device)
            )
            total_error = view1_error + view2_error
        return total_error.item()

    def transform(self, Xs):
        r"""
        Transform the given view with the trained autoencoder. Provide
        a single view within a list.

        Parameters
        ----------
        Xs : a list of exactly one array-like, or an np.ndarray
            Represents the View1 of some data. The array must have the same
            number of columns  (features) as the View1 presented
            in the :code:`fit(...)` step.
             - Xs length: 1
             - Xs[0] shape: (n_samples, n_features_0)

        Returns
        ----------
        embedding : np.ndarray of shape (n_samples, embedding_size)
            the embedding of the View1 data
        view1_reconstructions : np.ndarray of shape (n_samples, n_features_0)
            the reconstructed View1
        view2_prediction : np.ndarray of shape (n_samples, n_features_1)
            the predicted View2
        """
        Xs = check_Xs(Xs, enforce_views=1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        view1 = torch.FloatTensor(Xs[0])
        with torch.no_grad():
            embedding = self.view1_encoder_(view1.to(device))
            view1_reconstruction = self.view1_decoder_(embedding)
            view2_prediction = self.view2_decoder_(embedding)
        return (
            embedding.cpu().numpy(),
            view1_reconstruction.cpu().numpy(),
            view2_prediction.cpu().numpy(),
        )

    def fit_transform(self, Xs, y=None):
        r"""
        :code:`fit(Xs)` and then :code:`transform(Xs[:1])`.
        Note that this method will be
        embedding data that the autoencoder was trained on.

        Parameters
        ----------
        Xs : see :code:`fit(...)` Xs parameters

        y : ignored
            Included for API compliance.

        Returns
        ----------
        See :code:`transform(...)` return values.
        """
        self.fit(Xs)
        return self.transform(Xs[:1])
