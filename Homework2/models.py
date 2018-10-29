import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm


class KMeans:
    def __init__(self, n_clusters=4, init='k-means++', max_iter=300, epsilon=10**(-4)):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = np.array([])
        self.epsilon = epsilon
        self.labels = np.zeros(2)

    def fit(self, x):
        if self.init == 'k-means++':
            centroids = np.array([np.random.randint(len(x))])
            for i in range(1,self.n_clusters):
                proba = np.square(sm.pairwise_distances_argmin_min(X, centroids))
                proba /= proba.sum()
                centroids = np.append(centroids, x[np.random.choice(len(x), p=proba)].reshape(1,x.shape[1]))
        else :
            centroids = x[np.random.randint(len(x), size=self.n_clusters)]
        step = 0
        distortion = 100
        new_distortion = 0
        while step < self.max_iter and (np.abs(new_distortion-distortion) < self.epsilon):
            labels = sm.pairwise_distances_argmin(x,centroids)
            centroids = np.array([x[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            new_distortion = sum([sm.pairwise_distances(x[c==i], centroids[i]) for i in range(self.n_clusters)])
            step+=1
        self.centroids= centroids
        self.labels = sm.pairwise_distances_argmin(x, self.centroids)

    def predict(self, x):
        return sm.pairwise_distances_argmin(x, self.centroids)

class EM:
    def __init__(self, n_clusters=4, max_iter=300, epsilon=10**(-4)):
        self.n_clusters=n_clusters
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.mus = np.zeros((n_clusters,2))
        self.sigmas = np.zeros((n_clusters,2,2))
        self.latent = np.zeros((2,self.n_clusters))
        self.pi = np.zeros(self.n_clusters)

    def fit(self, x, circle=False):
        kmean = KMeans(self.n_clusters)
        kmean.fit(X)
        mus = kmean.centroids
        x_reduced = np.array([ x[kmean.labels == i] - mus[i] for i in range(self.n_clusters)])
        sigmas = np.array([ x_reduced.T @ x_reduced for i in range(self.n_clusters)])
        latent = np.zeros((len(x),self.n_clusters))
        pi = np.array([np.count_nonzero(kmean.labels == k) for k in range(self.n_clusters)]) / len(x)
        if circle:
            sigmas_c = np.array([ np.diag(s).mean() for s in sigmas])
            step = 0
            while step < self.max_iter:
                # E step
                for n,e in enumerate(x):
                    for k in range(self.n_clusters):
                        latent[n,k] = pi[k] * (1 / np.sqrt(sigmas_c[k]) * np.exp(-1/(2*sigmas_c[k]) * (x[n] - mus[k]).T @ (x[n] - mus[k])))
                    
                # M step

