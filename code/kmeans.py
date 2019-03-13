import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.cluster.vq import whiten

def run_kmeans_once(data, means, decel=True):
    distances = euclidean_distances(data, means)
    new_means = []
    for i in range(means.shape[0]):
        argmins = (distances.argmin(axis=-1) == i)
        new_means.append(data[argmins].mean(axis=0))
    change = (np.array(new_means) - means)
    lr = .1 if decel else 1.0
    means += lr * change

    distances = euclidean_distances(data, means)
    labels = distances.argmin(axis=-1)
    
    return means, labels

def kmeans_update(data, means):
    distances = euclidean_distances(data, means)
    new_means = []
    for i in range(means.shape[0]):
        assigned_to_mean = (distances.argmin(axis=-1) == i)
        new_means.append(data[assigned_to_mean].mean(axis=0))
    means = np.array(new_means)    
    return means