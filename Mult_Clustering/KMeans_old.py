import sklearn
from sklearn.datasets import fetch_20newsgroups, fetch_20newsgroups_vectorized
import numpy as np
import scipy as scp

NUM_SAMPLES = 200
LARGE_VAL = 10000000
ITER_THRESH = 5

def get_data():
    news_groups_all = fetch_20newsgroups(subset='all')
    news_data = news_groups_all.data

    #Load in the vectorized news group data from scikit-learn package
    vectorized_news = fetch_20newsgroups_vectorized(subset='all')
    all_data = vectorized_news.data
    all_targets = np.array(vectorized_news.target)
    class_names = vectorized_news.target_names

    #Set class pairings as described in the multiview clustering paper
    class_pairs = [['comp.graphics', 'rec.autos'],['rec.motorcycles', 'sci.med'], ['sci.space', 'misc.forsale'], ['rec.sport.hockey', 'soc.religion.christian'], ['comp.sys.ibm.pc.hardware', 'comp.os.ms-windows.misc']]

    #Create lists to hold data and labels for each of the 5 class across 2 different views
    labels =  [num for num in range(len(class_pairs)) for _ in range(NUM_SAMPLES)]
    labels = np.array(labels)
    view1_data = list()
    view2_data = list()
    views_data = [view1_data, view2_data]
    
    #Randomly sample 200 items from each of the selected classes in each pair
    for ind1 in range(len(class_pairs)):
        for ind2 in range(len(class_pairs[0])):
            class_num = class_names.index(class_pairs[ind1][ind2])
            class_data = all_data[(all_targets == class_num)]
            indices = np.random.choice(class_data.shape[0], NUM_SAMPLES)
            views_data[ind2].append(class_data[indices])

    #Shuffle and normalize vectors
    shuffled_inds = np.random.permutation(NUM_SAMPLES * len(class_pairs))
    view1_data = sparse.vstack(view1_data)
    view2_data = sparse.vstack(view2_data)
    view1_data = np.array(view1_data[shuffled_inds].todense())
    view2_data = np.array(view2_data[shuffled_inds].todense())
    magnitudes1 = np.linalg.norm(view1_data, axis=1)
    magnitudes2 = np.linalg.norm(view2_data, axis=1)
    magnitudes1[magnitudes1 == 0] = 1
    magnitudes2[magnitudes2 == 0] = 1
    magnitudes1 = magnitudes1.reshape((-1,1))
    magnitudes2 = magnitudes2.reshape((-1,1))
    view1_data /= magnitudes1
    view2_data /= magnitudes2
    labels = labels[shuffled_inds]
    
    return view1_data, view2_data, labels

def initialize_partitions(data, c_centers):
    cosine_sims = np.matmul(data, np.transpose(c_centers))
    new_partitions = np.argmax(cosine_sims,axis = 1).flatten()
    return new_partitions
    
def compute_objective(data, c_centers, partitions):

    o_funct = 0
    for clust in range(c_centers.shape[0]):
        vecs = data[(partitions == clust)]
        dot_products = np.matmul(vecs, c_centers[clust].transpose())
        clust_sum = np.sum(dot_products)
        o_funct += clust_sum
    return o_funct
    
def iterate_clusters(data, c_centers, partitions):

    #Recompute cluster centers
    new_centers = list()
    for clust in range(c_centers.shape[0]):
        summed_vec = np.sum(data[(partitions == clust)], axis = 0)
        vec = summed_vec / np.linalg.norm(summed_vec)
        new_centers.append(vec)
    new_centers = np.vstack(new_centers)
        
    #Assign each sample point to a partition
    cosine_sims = np.matmul(data, np.transpose(new_centers))
    new_partitions = np.argmax(cosine_sims,axis = 1).flatten()

    return new_centers, new_partitions

def compute_entropy():
    

def spherical_kmeans(v1_data, v2_data, labels, k = 6):

    #Initialize cluster centers, partitions, and loop params
    c_centers1 = np.random.random((k, v1_data.shape[1]))
    c_centers2 = np.random.random((k, v1_data.shape[1]))
    c_centers1 /= np.linalg.norm(c_centers1, axis=1).reshape((-1, 1))
    c_centers2 /= np.linalg.norm(c_centers2, axis=1).reshape((-1, 1))
    partitions1 = initialize_partitions(v1_data, c_centers1)
    partitions2 = initialize_partitions(v2_data, c_centers2)
    objective = [0, 0]
    iter_stall = 0

    while(iter_stall < ITER_THRESH):
        
        #Switch partitions, Maximization, and Expectation
        part_bucket = partitions1
        partitions1 = partitions2
        partitions2 = part_bucket
        c_centers1, partitions1 = iterate_clusters(v1_data, c_centers1, partitions1)
        c_centers2, partitions2 = iterate_clusters(v2_data, c_centers2, partitions2)
        o_funct1 = compute_objective(v1_data, c_centers1, partitions1)
        o_funct2 = compute_objective(v2_data, c_centers2, partitions2)
        iter_stall += 1
        
        #Recompute objective function
        if(o_funct1 > objective[0]):
            objective[0] = o_funct1
            iter_stall = 0
        if(o_funct2 > objective[1]):
            objective[1] = o_funct2
            iter_stall = 0
        print(objective)

    #Final cluster assignments
    
def main():
    v1_data, v2_data, labels = get_data()
    spherical_kmeans(v1_data, v2_data, labels)
    
if __name__ == "__main__":
    main()
