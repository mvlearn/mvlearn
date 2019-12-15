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


class FullyConnectedNet(torch.nn.Module):
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
    hiddenSize: number of nodes in the hidden layers
    numHiddenLayers: number of hidden layers in each encoder or
        decoder net
    embedSize: size of the bottleneck vector in the autoencoder
    trainingEpochs: how many times the network trains on the full
        dataset
    learningRate: learning rate of the Adam optimizer
    Attributes
    ----------
    view1Encoder: the View1 embedding network as a PyTorch module
    view1Decoder: the View1 decoding network as a PyTorch module
    view2Decoder: the View2 decoding network as a PyTorch module
    """

    def __init__(self, hiddenSize=64, numHiddenLayers=2, embedSize=20,
                 trainingEpochs=10, batchSize=16, learningRate=0.001):
        self.hiddenSize = hiddenSize
        self.embedSize = embedSize
        self.numHiddenLayers = numHiddenLayers
        self.trainingEpochs = trainingEpochs
        self.batchSize = batchSize
        self.learningRate = learningRate

    # Xs is not a tensor but instead a list with two arrays of shape [n, f_i]
    def fit(self, Xs, validationXs=None, printInfo=True):
        """
        Given two views, create and train the autoencoder.
        Parameters
        ----------
        Xs : list of array-likes or numpy.ndarray. Xs[0] is View1 and
        Xs[1] is View2
             - Xs length: n_views, only 2 is currently supported for splitAE.
             - Xs[i] shape: (n_samples, n_features_i)
        """

        Xs = check_Xs(Xs, multiview=True, enforce_views=2)
        assert Xs[0].shape[0] >= self.batchSize, """batch size must be <= to
            number of samples"""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        view1 = torch.FloatTensor(Xs[0])
        view2 = torch.FloatTensor(Xs[1])

        self.view1Encoder = FullyConnectedNet(view1.shape[1], self.hiddenSize,
                                              self.numHiddenLayers,
                                              self.embedSize).to(device)
        self.view1Decoder = FullyConnectedNet(self.embedSize, self.hiddenSize,
                                              self.numHiddenLayers,
                                              view1.shape[1]).to(device)
        self.view2Decoder = FullyConnectedNet(self.embedSize, self.hiddenSize,
                                              self.numHiddenLayers,
                                              view2.shape[1]).to(device)

        if printInfo:
            print("Parameter counts: \nview1Encoder: {:,}\nview1Decoder: {:,}"
                  "\nview2Decoder: {:,}".format(self.view1Encoder.paramCount(),
                                                self.view1Decoder.paramCount(),
                                                self.view2Decoder.paramCount())
                  )

        parameters = [self.view1Encoder.parameters(),
                      self.view1Decoder.parameters(),
                      self.view2Decoder.parameters()]
        optim = torch.optim.Adam(itertools.chain(*parameters),
                                 lr=self.learningRate)
        nSamples = view1.shape[0]
        epochTrainErrors = []
        epochTestErrors = []

        for epoch in tqdm.tqdm(range(self.trainingEpochs),
                               disable=(not printInfo)):
            batchErrors = []
            for batchNum in range(nSamples // self.batchSize):
                optim.zero_grad()
                view1Batch = view1[batchNum*self.batchSize:
                                   (batchNum+1)*self.batchSize]
                view2Batch = view2[batchNum*self.batchSize:
                                   (batchNum+1)*self.batchSize]
                embedding = self.view1Encoder(view1Batch.to(device))
                view1Reconstruction = self.view1Decoder(embedding)
                view2Reconstruction = self.view2Decoder(embedding)
                view1Error = torch.nn.MSELoss()(view1Reconstruction,
                                                view1Batch.to(device))
                view2Error = torch.nn.MSELoss()(view2Reconstruction,
                                                view2Batch.to(device))
                totalError = view1Error + view2Error
                totalError.backward()
                optim.step()
                batchErrors.append(totalError.item())
            print("Average train error during epoch {} was {}"
                  .format(epoch, np.mean(batchErrors))) if printInfo else None
            epochTrainErrors.append(np.mean(batchErrors))
            if validationXs is not None:
                testError = self._testError(validationXs)
                print("Average test  error during epoch {} was {}\n"
                      .format(epoch, testError)) if printInfo else None
                epochTestErrors.append(testError)

        if printInfo:
            plt.plot(epochTrainErrors, label="train error")
            if validationXs is not None:
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
            embedding = self.view1Encoder(view1Batch.to(device))
            view1Reconstruction = self.view1Decoder(embedding)
            view2Reconstruction = self.view2Decoder(embedding)
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
        Xs: a list of one array-like, or an np.ndarray, representing the
        View1 of some data. The array must have the same number of
        columns  (features) as the View1 presented in the `fit(...)` step.
             - Xs length: 1
             - Xs[0] shape: (n_samples, n_features_0)
        Returns
        ----------
        embedding: the embedding of the View1 data
        view1Reconstruction: the reconstructed View1
        view2Prediction: the predicted View2
        """
        Xs = check_Xs(Xs, enforce_views=1)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        view1 = torch.FloatTensor(Xs[0])
        with torch.no_grad():
            embedding = self.view1Encoder(view1.to(device))
            view1Reconstruction = self.view1Decoder(embedding)
            view2Prediction = self.view2Decoder(embedding)
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
