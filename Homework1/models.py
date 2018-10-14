import numpy as np
import matplotlib.pyplot as plt


class LDA:
    def __init__(self):
        self.sigma = 0
        self.inv_sigma = 0
        self.pi = 0
        self.mu0 = 0
        self.mu1 = 0
        self.w = 0
        self.alpha = 0
        self.fitted = False

    def fit(self, x, y):
        N = len(x)
        n = np.sum(y == 0)
        self.pi = n / N
        self.mu0 = np.mean(x[y == 0], axis=0)
        self.mu1 = np.mean(x[y == 1], axis=0)
        self.sigma = (np.sum([np.outer(xi - self.mu0, xi - self.mu0) for xi in x[y == 0]], axis=0) + np.sum(
            [np.outer(xi - self.mu1, xi - self.mu1) for xi in x[y == 1]], axis=0)) / N
        self.inv_sigma = np.linalg.inv(self.sigma)
        self.w = - np.dot(self.inv_sigma, self.mu1 - self.mu0)
        self.alpha = self.pi / (1 - self.pi) * np.exp(.5 * (
                np.dot(self.mu1.T, np.dot(self.inv_sigma, self.mu1)) - np.dot(self.mu0.T,
                                                                              np.dot(self.inv_sigma, self.mu0))))
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise NameError('Not fitted')
        tmp = - np.log(self.alpha) - np.dot(x, self.w)
        predictions = np.zeros(len(x))
        predictions[tmp > 0] = 1
        return predictions

    def score(self, x, y):
        return np.count_nonzero(self.predict(x) == y) / len(y)

    def plot(self, x, y):
        a = np.linspace(0, 10, 1000)
        sep = lambda x: (- np.log(self.alpha) - self.w[0] * x) / self.w[1]
        plt.plot(x=a, y=sep(a), color='black')
        plt.scatter(x[:, 0], x[:, 1], c=y)
        plt.show()
        print(sep(a))
        plt.show()
