from multiview.embed.splitae import SplitAE
import numpy as np

def test_splitae_overfit():
    nSamples = 10
    # give view1 10 features
    view1 = np.random.randn(nSamples, 10)
    # give view2 5 features, each of which is the sum of two features in view1
    view2 = view1[:, :5] + view1[:, -5:]
    # make huge network so we should overfit
    splitae = SplitAE(hiddenSize=64, numHiddenLayers=1, embedSize=10, trainingEpochs=200, batchSize=10, learningRate=0.01)
    splitae.fit([view1, view2], printInfo=False)
    embedding, reconstructedView1, predictedView2 = splitae.transform([view1])
    # std reaches 0.0035 max in 100 runs.
    assert np.std(reconstructedView1 - view1) < 1e-2
    assert np.std(predictedView2 - view2) < 1e-2
