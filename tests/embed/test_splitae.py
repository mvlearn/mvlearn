from multiview.embed.splitae import SplitAE
import numpy as np
import sklearn.cluster
import sklearn.datasets
from itertools import permutations
import torch

# These are blobs that PCA can reduce to nice clusters easily, so splitAE should be able to aswell
# cluster_std is high s.t. any 2 features alone do not make nice seperable blobs
def test_splitae_blobs():
    accuracies = []
    for i in range(10):
        view1, labels = sklearn.datasets.make_blobs(n_samples=1000, n_features=20, centers=None, cluster_std=10.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)
        # plt.scatter(*view1.T[2:4], c=labels)
        linear = torch.nn.Linear(20, 20)
        view2 = linear(torch.FloatTensor(view1)).detach().cpu() # view2 is just linear transform of view1
        trainSplit = 800

        # 0 hidden layers -- i.e. no nonlinearities, just matrix mults -- can solve this problem
        splitae = SplitAE(hiddenSize=64, numHiddenLayers=0, embedSize=2, trainingEpochs=10, batchSize=10, learningRate=0.01, printInfo=False, printGraph=False)
        splitae.fit([view1[:trainSplit], view2[:trainSplit]], validationXs=[view1[trainSplit:], view2[trainSplit:]])

        embeddings, reconstructedView1, predictedView2 = splitae.transform([view1[-200:]])
        # plt.scatter(*embeddings.T, c=labels[trainSplit:])
        kmeans = sklearn.cluster.KMeans(n_clusters=3).fit(embeddings)
        # plt.scatter(*embeddings.T, c=(kmeans.labels_))

        def clusterAccuracy(prediction, target, nClusters):
            predictionPermuted = prediction.copy()
            clusterPermutations = list(permutations(range(nClusters)))
            maxAccuracy = 0
            for permutation in clusterPermutations:
                indexes = []
                for i in range(nClusters):
                    indexes.append(prediction == i)
                for i in range(nClusters):
                    predictionPermuted[indexes[i]] = permutation[i]
                accuracy = np.sum(predictionPermuted == target) / len(target)
                maxAccuracy = accuracy if accuracy > maxAccuracy else maxAccuracy
            return maxAccuracy

        accuracy = clusterAccuracy(kmeans.labels_, labels[trainSplit:], nClusters=3)
        accuracies.append(accuracy)
    # distribution of accuracies sometimes has outliers around 0.65, so be conservative
    assert np.mean(accuracies) > 0.7

def test_splitae_overfit():
    nSamples = 10
    # give view1 10 features
    view1 = np.random.randn(nSamples, 10)
    # give view2 5 features, each of which is the sum of two features in view1
    view2 = view1[:, :5] + view1[:, -5:]
    # make huge network so we should overfit
    splitae = SplitAE(hiddenSize=64, numHiddenLayers=1, embedSize=10, trainingEpochs=200, batchSize=10, learningRate=0.01, printInfo=False, printGraph=False)
    # irrelevant validationXs to make sure testing error code runs
    splitae.fit([view1, view2], validationXs=[view1, view2])
    embedding, reconstructedView1, predictedView2 = splitae.transform([view1])
    # thresholds picked by looking at distributions of these errors
    assert np.mean(reconstructedView1 - view1) < 1e-3
    assert np.mean(predictedView2 - view2) < 1e-3
    assert np.std(reconstructedView1 - view1) < 4e-2
    assert np.std(predictedView2 - view2) < 4e-2
