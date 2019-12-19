import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import PIL
import numpy as np
import itertools
import tqdm

from .base import BaseEmbed
from ..utils.utils import check_Xs


class _FullyConnectedNet(torch.nn.Module):
    """
    General torch module for a fully connected neural network with
    an input layer with # nodes given by inputSize , numHiddenLayers hidden
    layers with # nodes given by hiddenSize, and an output layer # nodes given
    by embeddingSize.
    """
    def __init__(self, inputSize, hiddenSize, numHiddenLayers, embeddingSize):
        super().__init__()
        assert numHiddenLayers >= 0, "can't have negative hidden layer count"
        self.layers = torch.nn.ModuleList()
        if numHiddenLayers == 0:
            self.layers.append(torch.nn.Linear(inputSize, embeddingSize))
        else:
            self.layers.append(torch.nn.Linear(inputSize, hiddenSize))
            for i in range(numHiddenLayers-1):
                self.layers.append(torch.nn.Linear(hiddenSize, hiddenSize))
            self.layers.append(torch.nn.Linear(hiddenSize, embeddingSize))

    def forward(self, x):
        # Forward pass for the network. Pytorch automatically calculates
        # backwards pass
        for layer in self.layers[:-1]:
            x = torch.nn.Sigmoid()(layer(x))
        x = self.layers[-1](x)  # no activation on last layer
        return x

    def paramCount(self):
        return np.sum([np.prod(s.shape) for s in self.parameters()])


class SplitAE(BaseEmbed):
    """
    Implements an autoencoder that creates an embedding of a view View1 and
    from that embedding reconstructs View1 and another view View2.

    Parameters
    ----------
    hidden_size: int (default=64)
        number of nodes in the hidden layers
    num_hidden_layers: int (default=2)
        number of hidden layers in each encoder or decoder net
    embed_size: int (default=20)
        size of the bottleneck vector in the autoencoder
    training_epochs: int (default=10)
        how many times the network trains on the full dataset
    batch_size: int (default=16):
        batch size while training the network
    learning_rate: float (default=0.001)
        learning rate of the Adam optimizer
    print_info: bool (default=True)
        whether or not to print errors as the network trains.
    print_graph: bool (default=True)
        whether or not to graph training loss

    Attributes
    ----------
    view1_encoder_: torch.nn.Module
        the View1 embedding network as a PyTorch module
    view1_decoder_: torch.nn.Module
        the View1 decoding network as a PyTorch module
    view2_decoder_: torch.nn.Module
        the View2 decoding network as a PyTorch module
    """

    def __init__(self, hidden_size=64, num_hidden_layers=2, embed_size=20,
                 training_epochs=10, batch_size=16, learning_rate=0.001,
                 print_info=False, print_graph=True):
        self.hiddenSize = hidden_size
        self.embedSize = embed_size
        self.numHiddenLayers = num_hidden_layers
        self.trainingEpochs = training_epochs
        self.batchSize = batch_size
        self.learningRate = learning_rate
        self.printInfo = print_info
        self.printGraph = print_graph

    # Xs is not a tensor but instead a list with two arrays of shape [n, f_i]
    def fit(self, Xs, validation_Xs=None):
        """
        Given two views, create and train the autoencoder.

        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray.
            Xs[0] is View1 and Xs[1] is View2
             - Xs length: n_views, only 2 is currently supported for splitAE.
             - Xs[i] shape: (n_samples, n_features_i)
        validation_Xs: list of array-likes or numpy.ndarray
            optional validation data in the same shape of Xs. If
            printInfo is true, then validation error, calculated with this
            data, will be printed as the network trains.
        """

        Xs = check_Xs(Xs, multiview=True, enforce_views=2)
        assert Xs[0].shape[0] >= self.batchSize, """batch size must be <= to
            number of samples"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        view1 = torch.FloatTensor(Xs[0])
        view2 = torch.FloatTensor(Xs[1])

        self.view1Encoder_ = _FullyConnectedNet(view1.shape[1],
                                                self.hiddenSize,
                                                self.numHiddenLayers,
                                                self.embedSize).to(device)
        self.view1Decoder_ = _FullyConnectedNet(self.embedSize,
                                                self.hiddenSize,
                                                self.numHiddenLayers,
                                                view1.shape[1]).to(device)
        self.view2Decoder_ = _FullyConnectedNet(self.embedSize,
                                                self.hiddenSize,
                                                self.numHiddenLayers,
                                                view2.shape[1]).to(device)

        self.view1_encoder_ = self.view1Encoder_
        self.view1_decoder_ = self.view1Decoder_
        self.view2_decoder_ = self.view2Decoder_

        if self.printGraph:
            print("Parameter counts: \nview1Encoder: {:,}\nview1Decoder: {:,}"
                  "\nview2Decoder: {:,}"
                  .format(self.view1Encoder_.paramCount(),
                          self.view1Decoder_.paramCount(),
                          self.view2Decoder_.paramCount())
                  )

        parameters = [self.view1Encoder_.parameters(),
                      self.view1Decoder_.parameters(),
                      self.view2Decoder_.parameters()]
        optim = torch.optim.Adam(itertools.chain(*parameters),
                                 lr=self.learningRate)
        nSamples = view1.shape[0]
        epochTrainErrors = []
        epochTestErrors = []

        for epoch in tqdm.tqdm(range(self.trainingEpochs),
                               disable=(not self.printInfo)):
            batchErrors = []
            for batchNum in range(nSamples // self.batchSize):
                optim.zero_grad()
                view1Batch = view1[batchNum*self.batchSize:
                                   (batchNum+1)*self.batchSize]
                view2Batch = view2[batchNum*self.batchSize:
                                   (batchNum+1)*self.batchSize]
                embedding = self.view1Encoder_(view1Batch.to(device))
                view1Reconstruction = self.view1Decoder_(embedding)
                view2Reconstruction = self.view2Decoder_(embedding)
                view1Error = torch.nn.MSELoss()(view1Reconstruction,
                                                view1Batch.to(device))
                view2Error = torch.nn.MSELoss()(view2Reconstruction,
                                                view2Batch.to(device))
                totalError = view1Error + view2Error
                totalError.backward()
                optim.step()
                batchErrors.append(totalError.item())
            if self.printInfo:
                print("Average train error during epoch {} was {}"
                      .format(epoch, np.mean(batchErrors)))
            epochTrainErrors.append(np.mean(batchErrors))
            if validation_Xs is not None:
                testError = self._testError(validation_Xs)
                if self.printInfo:
                    print("Average test  error during epoch {} was {}\n"
                          .format(epoch, testError))
                epochTestErrors.append(testError)

        if self.printGraph:
            plt.plot(epochTrainErrors, label="train error")
            if validation_Xs is not None:
                plt.plot(epochTestErrors, label="test error")
            plt.title("Errors during training")
            plt.xlabel("Epoch")
            plt.ylabel("Error")
            plt.legend()
            plt.show()

    def _testError(self, Xs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        nSamples = Xs[0].shape[0]
        validationBatchSize = self.batchSize
        testIndices = np.random.choice(nSamples, validationBatchSize,
                                       replace=False)
        view1Batch = torch.FloatTensor(Xs[0][testIndices])
        view2Batch = torch.FloatTensor(Xs[1][testIndices])
        with torch.no_grad():
            embedding = self.view1Encoder_(view1Batch.to(device))
            view1Reconstruction = self.view1Decoder_(embedding)
            view2Reconstruction = self.view2Decoder_(embedding)
            view1Error = torch.nn.MSELoss()(view1Reconstruction,
                                            view1Batch.to(device))
            view2Error = torch.nn.MSELoss()(view2Reconstruction,
                                            view2Batch.to(device))
            totalError = view1Error + view2Error
        return totalError.item()

    def transform(self, Xs):
        """
        Transform the given view with the trained autoencoder.

        Parameters
        ----------
        Xs: a list of one array-like, or an np.ndarray
            Represents the View1 of some data. The array must have the same
            number of columns  (features) as the View1 presented
            in the `fit(...)` step.
             - Xs length: 1
             - Xs[0] shape: (n_samples, n_features_0)

        Returns
        ----------
        embedding: np.ndarray of shape (n_samples, embeddingSize)
            the embedding of the View1 data
        view1Reconstruction: np.ndarray of shape (n_samples, n_features_0)
            the reconstructed View1
        view2Prediction: np.ndarray of shape (n_samples, n_features_1)
            the predicted View2
        """
        Xs = check_Xs(Xs, enforce_views=1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        view1 = torch.FloatTensor(Xs[0])
        with torch.no_grad():
            embedding = self.view1Encoder_(view1.to(device))
            view1Reconstruction = self.view1Decoder_(embedding)
            view2Prediction = self.view2Decoder_(embedding)
        return (embedding.cpu().numpy(), view1Reconstruction.cpu().numpy(),
                view2Prediction.cpu().numpy())

    def fit_transform(self, Xs):
        """
        `fit(Xs)` and then `transform(Xs[:1])`. Note that this method will be
        embedding data that the autoencoder was trained on.

        Parameters:
        ----------
        Xs: see `fit(...)` Xs parameters

        Returns
        ----------
        See `transform(...)` return values.
        """
        self.fit(Xs)
        return self.transform(Xs[:1])
