import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import seaborn as sns
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(flatui))


def load_file(file_name):
    def convert(line):
        x1, x2 = line.split()
        return np.array([np.float(x1), np.float(x2)])

    with open(file_name) as f:
        content = np.array([convert(line) for line in f])
    return np.array(content)


class KMeans:
    def __init__(self, n_clusters=4, init='k-means++', max_iter=300, epsilon=10 ** (-4)):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.cluster_centers_ = np.array([])
        self.epsilon = epsilon
        self.labels_ = np.zeros(2)

    def fit(self, x, return_distortion=False):
        if self.init == 'k-means++':
            centroids = x[[np.random.randint(len(x))]]
            for i in range(1, self.n_clusters):
                proba = np.square(sm.pairwise_distances_argmin_min(x, centroids)[1])
                proba /= proba.sum()
                centroids = np.append(centroids, x[np.random.choice(len(x), p=proba)].reshape(1, x.shape[1]), axis=0)
        else:
            centroids = x[np.random.randint(len(x), size=self.n_clusters)]
        step = 0
        change = True
        while step < self.max_iter and change:
            labels = sm.pairwise_distances_argmin(x, centroids)
            centroids = np.array([x[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            change = not (labels == sm.pairwise_distances_argmin(x, centroids)).all()
            step += 1
        self.cluster_centers_ = centroids
        self.labels_ = sm.pairwise_distances_argmin(x, self.cluster_centers_)
        if return_distortion:
            dif = x - centroids[self.labels_]
            return np.einsum('ij,ij->i', dif, dif).sum()
        return None

    def fit_predict(self, x):
        self.fit(x)
        return self.labels_

    def predict(self, x):
        return sm.pairwise_distances_argmin(x, self.cluster_centers_)

    def plot(self, x, labels=None):
        if labels is None:
            labels = self.labels_
        palette = sns.color_palette("husl", n_colors=self.n_clusters)
        sns.scatterplot(x=x[:,0], y=x[:,1], hue=labels, palette=palette)
        plt.scatter(x=self.cluster_centers_[:,0], y=self.cluster_centers_[:,1], color='black')
        plt.show()



class EM:
    def __init__(self, n_clusters=4, max_iter=300, epsilon=10 ** (-4)):
        self.n_clusters = n_clusters
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.mus = np.zeros((n_clusters, 2))
        self.sigmas = np.zeros((n_clusters, 2, 2))
        self.latent = np.zeros((2, self.n_clusters))
        self.pi = np.zeros(self.n_clusters)
        self.fitted = False
        self.sphere = False
        self.x = None
        self.labels = None

    def fit(self, x, sphere=True):
        self.sphere = sphere
        self.x = x
        n, d = x.shape
        plt.scatter(x=x[:, 0], y=x[:, 1])
        plt.plot()
        kmean = KMeans(self.n_clusters)
        kmean.fit(x)
        labels = kmean.labels_
        mus = kmean.cluster_centers_
        x_reduced = x - mus[kmean.labels_]
        latent = np.zeros((len(x), self.n_clusters))
        pi = np.array([np.count_nonzero(kmean.labels_ == k) for k in range(self.n_clusters)]) / len(x)
        j = 0
        update = True
        if sphere:
            sigmas = np.array(
                [np.sqrt(np.diag(np.cov(x_reduced[kmean.labels_ == i].T))).mean() for i in range(self.n_clusters)])
            while j < self.max_iter and True:
                # E step: update latent
                latent = np.array([pi[i] * np.exp(-np.square(np.linalg.norm(x - mus[i], axis=1))/(2*sigmas[i]**2))/sigmas[i]**d for i in range(self.n_clusters)])
                latent /= latent.sum(axis=0)
                new_labels = np.argmax(latent, axis=0)
                update = not (labels==new_labels).all()
                labels = new_labels
                # M step: update pi, mus and sigma
                pi = latent.sum(axis=1)/n
                mus = (latent @ x) / latent.sum(axis=1).reshape(4,1)
                sigmas = np.array([latent[k] @ np.square(np.linalg.norm(x - mus[k], axis=1)) for k in range(self.n_clusters)]) / (d * latent.sum(axis=1))
                sigmas = np.sqrt(sigmas)
                j +=1
        else:
            sigmas = np.array([np.array(np.cov(x_reduced[kmean.labels_ == i].T)) for i in range(self.n_clusters)])
            while j < self.max_iter and True:
                # E step: update latent
                inv_sigmas = np.linalg.inv(sigmas)
                det_sigmas = np.linalg.det(sigmas)
                latent = np.array([pi[i] * np.exp(- 0.5 * np.einsum('ij,ij->i', (x - mus[i]) @ inv_sigmas[i], x - mus[i]))/np.sqrt(det_sigmas[i]) for i in range(self.n_clusters)])
                latent /= latent.sum(axis=0)
                new_labels = np.argmax(latent, axis=0)
                update = not (labels==new_labels).all()
                labels = new_labels
                # M step: update pi, mus and sigma
                pi = latent.sum(axis=1)/n
                mus = (latent @ x) / latent.sum(axis=1).reshape(4,1)
                sigmas = np.array([(latent[k].reshape(500,1) * (x - mus[k])).T @ (x - mus[k]) / latent[k].sum() for k in range(self.n_clusters)])
                j +=1
        self.pi = pi
        self.mus = mus
        self.latent =latent
        self.sigmas = sigmas
        self.labels = labels
        self.fitted = True

    def plot(self):
        if not self.fitted:
            raise NameError('Not fitted')
        palette = sns.color_palette("husl", n_colors=self.n_clusters)
        sns.scatterplot(x=self.x[:, 0], y=self.x[:, 1], hue=self.labels, palette=palette)
        plt.scatter(x=self.mus[:, 0], y=self.mus[:, 1], color='black')
        plt.show()
