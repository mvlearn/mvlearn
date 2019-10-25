import numpy as np
from pymix import mixture
import sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import OrdinalEncoder
import scipy as scp
from scipy import sparse

NUM_SAMPLES = 200

#A function to get the 20 newsgroup data
def get_data():
    
    #Load in the vectorized news group data from scikit-learn package
    news = fetch_20newsgroups(subset='all')
    all_data = np.array(news.data)
    preproc = CountVectorizer().build_preprocessor()
    data = list()
    for ind in range(all_data.shape[0]):
        data.append(all_data[ind].strip().split()[:100])
    all_data = np.array(data)
    print(all_data[1])
    print(all_data.shape)
    return
    all_targets = np.array(news.target)
    class_names = news.target_names
    
    #Set class pairings as described in the multiview clustering paper
    view1_classes = ['comp.graphics','rec.motorcycles', 'sci.space', 'rec.sport.hockey', 'comp.sys.ibm.pc.hardware']
    view2_classes = ['rec.autos', 'sci.med','misc.forsale', 'soc.religion.christian','comp.os.ms-windows.misc']
    
    #Create lists to hold data and labels for each of the 5 classes across 2 different views
    labels =  [num for num in range(len(view1_classes)) for _ in range(NUM_SAMPLES)]
    labels = np.array(labels)
    view1_data = list()
    view2_data = list()
    
    #Randomly sample 200 items from each of the selected classes in view1
    for ind in range(len(view1_classes)):
        class_num = class_names.index(view1_classes[ind])
        class_data = all_data[(all_targets == class_num)]
        indices = np.random.choice(class_data.shape[0], NUM_SAMPLES)
        view1_data.append(class_data[indices])
    view1_data = np.concatenate(view1_data)
        
        
        #Randomly sample 200 items from each of the selected classes in view2
    for ind in range(len(view2_classes)):
        class_num = class_names.index(view2_classes[ind])
        class_data = all_data[(all_targets == class_num)]
        indices = np.random.choice(class_data.shape[0], NUM_SAMPLES)
        view2_data.append(class_data[indices])
    view2_data = np.concatenate(view2_data)
    
    #Vectorize the data
    vectorizer = OrdinalEncoder()
    view1_data = vectorizer.fit_transform(view1_data)
    view2_data = vectorizer.fit_transform(view2_data)

    #Shuffle and normalize vectors
    shuffled_inds = np.random.permutation(NUM_SAMPLES * len(view1_classes))
    view1_data = sparse.vstack(view1_data)
    view2_data = sparse.vstack(view2_data)
    view1_data = np.array(view1_data[shuffled_inds].todense())
    view2_data = np.array(view2_data[shuffled_inds].todense())
    labels = labels[shuffled_inds]
    
    return view1_data, view2_data, labels


def main():
    view1_data, view2_data, labels = get_data()
    v_data = np.concatenate([view1_data, view2_data], axis = 1)
    #v_data = np.loadtxt('data.txt')
    #labels = np.loadtxt('labels.txt')
    data = mixture.DataSet()
    data.fromArray(v_data)
    np.savetxt('data.txt', v_data)
    np.savetxt('labels.txt', labels)

    print('what')
    
    components = list()
    probabilities = list(np.ones((v_data.shape[1],))/ v_data.shape[1])
    for ind in range(5):
        comp = mixture.MultinomialDistribution(100, v_data.shape[1], probabilities)
        components.append(comp)

    model = mixture.MixtureModel(5, [0.2, 0.2, 0.2, 0.2, 0.2], components)
    model.EM(data, 5, 0.1)
    clust = model.classify(data)
    
        
if __name__ == '__main__':
    main()
