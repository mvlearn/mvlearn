import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import itertools
import tqdm

from multiview.embed.base import BaseEmbed

class FullyConnectedNet(torch.nn.Module):
    def __init__(self, inputSize, hiddenSize, numHiddenLayers, embeddingSize):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(inputSize, hiddenSize))
        assert numHiddenLayers >= 0, "can't have negative hidden layer count"
        for i in range(numHiddenLayers):
            self.layers.append(torch.nn.Linear(hiddenSize, hiddenSize))
        self.layers.append(torch.nn.Linear(hiddenSize, embeddingSize))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.nn.Sigmoid()(layer(x))
        x = self.layers[-1](x) # no activation on last layer
        return x

    def paramCount(self):
        return np.sum([np.prod(s.shape) for s in self.parameters()])

class SplitAE(BaseEmbed):
    """
    Implements an autoencoder that creates an embedding of a view View1 and from that embedding reconstructs View1 and another view View2.
    Parameters
    ----------
    hiddenSize: number of nodes in the hidden layers
    numHiddenLayers: number of hidden layers in each encoder or
        decoder net
    embedSize: size of the bottleneck vector in the autoencoder
    trainingEpochs: how many times the network trains on the full
        dataset
    learningRate: learning rate of the Adam optimizer
    Attributes:
    ----------
    view1Encoder: the View1 embedding network as a PyTorch module
    view1Decoder: the View1 decoding network as a PyTorch module
    view2Decoder: the View2 decoding network as a PyTorch module
    """

    def __init__(self, hiddenSize=64, numHiddenLayers=2, embedSize=20, trainingEpochs=10, batchSize=16, learningRate=0.001):
        self.hiddenSize = hiddenSize
        self.embedSize = embedSize
        self.numHiddenLayers = numHiddenLayers
        self.trainingEpochs = trainingEpochs
        self.batchSize = batchSize
        self.learningRate = learningRate

    def fit(self, Xs): #Xs is not a tensor but instead a list with two arrays of shape [n, f_i]
        """
        Given two views, create and train the autoencoder.
        Parameters
        ----------
        Xs: a list with two arrays. Each array has `n` rows (samples) and some number of columns (features). The first array is View1 and the second array is View2.
        """

        # DATA FOR TESTING
        Xs = [torch.randn(1000, 20), torch.randn(1000, 30)]
        class self():
            hiddenSize = 100
            embedSize = 20
            numHiddenLayers = 2

        assert len(Xs) == 2, "this SplitAE implementation deals with two views"
        assert Xs[0].shape[0] == Xs[1].shape[0], "must have each view for each sample"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        view1 = torch.FloatTensor(Xs[0])
        view2 = torch.FloatTensor(Xs[1])

        self.view1Encoder = FullyConnectedNet(view1.shape[1], self.hiddenSize,
            self.numHiddenLayers, self.embedSize).to(device)
        self.view1Decoder = FullyConnectedNet(self.embedSize, self.hiddenSize,
            self.numHiddenLayers, view1.shape[1]).to(device)
        self.view2Decoder = FullyConnectedNet(self.embedSize, self.hiddenSize,
            self.numHiddenLayers, view2.shape[1]).to(device)

        print("Parameter counts: \nview1Encoder: {:,}\nview1Decoder: {:,}"
            "\nview2Decoder: {:,}".format(self.view1Encoder.paramCount(),
             self.view1Decoder.paramCount(), self.view2Decoder.paramCount()))

        parameters = [self.view1Encoder.parameters(), self.view1Decoder.parameters(), self.view2Decoder.parameters()]
        optim = torch.optim.Adam(itertools.chain(*parameters), lr=self.learningRate)
        nSamples = view1.shape[0]
        for epoch in range(trainingEpochs):
            errors = []
            for batchNum in tqdm.tqdm(range(nSamples // self.batchSize)):
                optim.zero_grad()
                view1Batch = view1[batchNum*self.batchSize:(batchNum+1)*self.batchSize]
                view2Batch = view2[batchNum*self.batchSize:(batchNum+1)*self.batchSize]
                embedding = self.view1Encoder(view1Batch.to(device))
                view1Reconstruction = self.view1Decoder(embedding)
                view2Reconstruction = self.view2Decoder(embedding)
                view1Error = torch.nn.MSELoss()(view1Reconstruction, view1Batch.to(device))
                view2Error = torch.nn.MSELoss()(view2Reconstruction, view2Batch.to(device))
                totalError = view1Error + view2Error
                totalError.backward()
                optim.step()
                errors.append(totalError.item())
            plt.plot(errors)
            print("Average reconstruction error during epoch {} was {}".format(epoch, np.mean(errors))

    def transform(self, Xs):
        """
        Transform the given view with the trained autoencoder.
        Parameters
        ----------
        Xs: a list with one array representing the View1 view of some data. The array must have the same number of columns (features) as the View1 presented in the `fit(...)` step.
        Returns
        ----------
        embedding: the embedding of the View1 data
        view1Reconstruction: the reconstructed View1
        view2Prediction: the predicted View2
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        view1 = torch.FloatTensor(Xs[0])
        with torch.no_grad():
            embedding = self.view1Encoder(view1.to(device))
            view1Reconstruction = self.view1Decoder(embedding)
            view2Prediction = self.view2Decoder(embedding)
        return (embedding.cpu().numpy(), view1Reconstruction.cpu().numpy(), view2Prediction.cpu().numpy())

    def fit_transform(self, Xs):
        """
        `fit(Xs)` and then `transform(Xs[:1])`. Note that this method will be embedding data that the autoencoder was trained on.
        Parameters:
        ----------
        Xs: see `fit(...)` Xs parameters
        Returns
        ----------
        See `transform(...)` return values.
        """
        self.fit(Xs)
        return self.transform(Xs[:1])



#-------------------- Will use below for proof-of-working / tutorial code (in seperate files) --------------------

#TODO: method that tests SplitAE for travis CI

# from plotly import offline as py
# import plotly.tools as tls
# py.init_notebook_mode()
# plotly = lambda: py.iplot(tls.mpl_to_plotly(plt.gcf())) #usage: put plotly() under plt.hist(...) and run the two lines together

# %matplotlib inline
# plt.style.use("ggplot")
# %config InlineBackend.figure_format = 'svg'
# np.set_printoptions(suppress=True) # don't use scientific [e.g. 5e10] notation


class NoisyMnist(Dataset):

    MNIST_MEAN, MNIST_STD = (0.1307, 0.3081)

    def __init__(self, train=True):
        super().__init__()
        self.mnistDataset = datasets.MNIST("./mnist", train=train, download=True)

    def __len__(self):
        return len(self.mnistDataset)

    def __getitem__(self, idx):
        randomIndex = lambda: np.random.randint(len(self.mnistDataset))
        image1, label1 = self.mnistDataset[idx]
        image2, label2 = self.mnistDataset[randomIndex()]
        while not label1 == label2:
            image2, label2 = self.mnistDataset[randomIndex()]

        image1 = torchvision.transforms.RandomRotation((-45, 45), resample=PIL.Image.BICUBIC)(image1)
        image2 = torchvision.transforms.RandomRotation((-45, 45), resample=PIL.Image.BICUBIC)(image2)
        image1 = np.array(image1) / 255
        image2 = np.array(image2) / 255

        image2 = np.clip(image2 + np.random.uniform(0, 1, size=image2.shape), 0, 1)

        image1 = (image1 - self.MNIST_MEAN) / self.MNIST_STD
        image2 = (image2 - (self.MNIST_MEAN+0.5-0.053)) / self.MNIST_STD

        image1 = torch.FloatTensor(image1).unsqueeze(0)
        image2 = torch.FloatTensor(image2).unsqueeze(0)

        return (image1, image2, label1)

from MulticoreTSNE import MulticoreTSNE as TSNE #sklearn TSNE too slow

testDataset = NoisyMnist(train=False)
testDataloader = DataLoader(testDataset, batch_size=10000, shuffle=True, num_workers=8)
with torch.no_grad():
    view1, view2, labels = next(iter(testDataloader))
    latents = encoder(view1.to(device))

pointColors = []
colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
origColors = [[55, 55, 55], [255, 34, 34], [38, 255, 38], [10, 10, 255], [255, 12, 255], [250, 200, 160], [120, 210, 180], [150, 180, 205], [210, 160, 210], [190, 190, 110]]
origColors = (np.array(origColors)) / 255
for l in labels.cpu().numpy():
    pointColors.append(tuple(origColors[l].tolist()))

tsne = TSNE(n_jobs=12)
tsneEmbeddings = tsne.fit_transform(latents.cpu().numpy())

tsneEmbeddingsNoEncode = tsne.fit_transform(view1.view(-1, 784).numpy())
tsneEmbeddingsNoEncodeNoisy = tsne.fit_transform(view2.view(-1, 784).numpy())
plt.scatter(*tsneEmbeddings.transpose(), c=pointColors, s=5)
plotly()
plt.scatter(*tsneEmbeddingsNoEncode.transpose(), c=pointColors, s=5)
plt.scatter(*tsneEmbeddingsNoEncodeNoisy.transpose(), c=pointColors, s=5)
