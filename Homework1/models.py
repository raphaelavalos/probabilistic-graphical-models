import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
        if not self.fitted:
            raise NameError('Not fitted')
        a = np.array([x[:, 0].min(), x[:, 0].max()])
        sep = lambda e: (- np.log(self.alpha) - self.w[0] * e) / self.w[1]
        df = pd.DataFrame()
        df['x1'] = x[:, 0]
        df['x2'] = x[:, 1]
        df['category'] = y
        fg = sns.lmplot(x='x1', y='x2', hue='category', data=df, fit_reg=False)
        fg.axes[0, 0].plot(a, sep(a))
        plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_sigmoid(x):
    return sigmoid(x) * sigmoid(-x)


class IRLS:
    def __init__(self, epsilon=10 ** (-4), max_step=10000):
        self.w = 0
        self.epsilon = epsilon
        self.max_step = max_step
        self.fitted = False

    def fit(self, x, y):
        X = np.c_[x, np.ones(len(x))]
        w = np.random.rand(np.shape(X)[1])
        update = [1000]
        step = 0
        while (step < self.max_step) and (np.linalg.norm(update, 2) > self.epsilon):
            eta = sigmoid(X @ w)
            update = np.linalg.inv(X.T @ np.diagflat(eta) @ X) @ X.T @ (y - eta)
            w += update
            step += 1
        self.w = w
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise NameError('Not fitted')
        X = np.c_[x, np.ones(len(x))]
        proba = sigmoid(X @ self.w)
        prediction = np.zeros(len(x))
        prediction[proba > .5] = 1
        return prediction

    def score(self, x, y):
        return np.count_nonzero(self.predict(x) == y) / len(y)

    def plot(self, x, y):
        if not self.fitted:
            raise NameError('Not fitted')
        a = np.array([x[:, 0].min(), x[:, 0].max()])
        sep = lambda e: -(self.w[2] + self.w[0] * e) / self.w[1]
        df = pd.DataFrame()
        df['x1'] = x[:, 0]
        df['x2'] = x[:, 1]
        df['category'] = y
        fg = sns.lmplot(x='x1', y='x2', hue='category', data=df, fit_reg=False)
        fg.axes[0, 0].plot(a, sep(a))
        plt.show()


class LinearRegression:
    def __init__(self):
        self.w = 0
        self.fitted = False

    def fit(self, x, y):
        X = np.c_[x, np.ones(len(x))]
        self.w = np.linalg.inv(X.T @ X) @ X.T @ y
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise NameError('Not fitted')
        X = np.c_[x, np.ones(len(x))]
        tmp = X @ self.w
        prediction = np.zeros(len(x))
        prediction[tmp > .5] = 1
        return prediction

    def score(self, x, y):
        return np.count_nonzero(self.predict(x) == y) / len(y)

    def plot(self, x, y):
        if not self.fitted:
            raise NameError('Not fitted')
        a = np.array([x[:, 0].min(), x[:, 0].max()])
        sep = lambda e: (.5 - self.w[2] - self.w[0] * e) / self.w[1]
        df = pd.DataFrame()
        df['x1'] = x[:, 0]
        df['x2'] = x[:, 1]
        df['category'] = y
        fg = sns.lmplot(x='x1', y='x2', hue='category', data=df, fit_reg=False)
        fg.axes[0, 0].plot(a, sep(a))
        plt.show()


class QDA:
    def __init__(self):
        self.sigma0 = 0
        self.sigma1 = 0
        self.inv_sigma0 = 0
        self.inv_sigma1 = 0
        self.pi = 0
        self.mu0 = 0
        self.mu1 = 0
        self.w = 0
        self.alpha = 0
        self.poly_on_y = lambda x: x
        self.fitted = False

    def fit(self, x, y):
        N = len(x)
        n = np.sum(y == 0)
        self.pi = n / N
        self.mu0 = np.mean(x[y == 0], axis=0)
        self.mu1 = np.mean(x[y == 1], axis=0)
        self.sigma0 = np.sum([np.outer(xi - self.mu0, xi - self.mu0) for xi in x[y == 0]], axis=0) / n
        self.sigma1 = np.sum([np.outer(xi - self.mu1, xi - self.mu1) for xi in x[y == 1]], axis=0) / (N - n)
        self.inv_sigma0 = np.linalg.inv(self.sigma0)
        self.inv_sigma1 = np.linalg.inv(self.sigma1)
        self.alpha = self.pi / (1 - self.pi) * np.sqrt(np.linalg.det(self.sigma0) / np.linalg.det(self.sigma1))
        pond_diff_inv_sigma = self.mu1.T @ self.inv_sigma1 - self.mu0.T @ self.inv_sigma0
        diff_inv_sigma = self.inv_sigma0 - self.inv_sigma1
        self.poly_on_y = lambda e: np.array(
            [.5 * diff_inv_sigma[1, 1], diff_inv_sigma[1, 0] * e + pond_diff_inv_sigma[1],
             .5 * diff_inv_sigma[0, 0] * e * e + pond_diff_inv_sigma[0] * e + np.exp(
                 self.alpha) - self.mu1.T @ self.inv_sigma1 @ self.mu1 + self.mu0.T @ self.inv_sigma0 @ self.mu0 ])
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise NameError('Not fitted')
        dif0 = x - self.mu0
        dif1 = x - self.mu1
        norm0 = .5 * np.einsum('ij,ij -> i', dif0 @ self.sigma0, dif0)
        norm1 = - .5 * np.einsum('ij,ij -> i', dif1 @ self.sigma1, dif1)
        tmp = np.log(self.alpha) + norm0 + norm1
        predictions = np.zeros(len(x))
        predictions[tmp > 0] = 1
        return predictions

    def score(self, x, y):
        return np.count_nonzero(self.predict(x) == y) / len(y)

    def plot(self, x, y):
        if not self.fitted:
            raise NameError('Not fitted')
        a = np.linspace(x[:, 0].min(), x[:, 0].max(), 100)
        e = np.array(list(map(lambda v: np.roots(self.poly_on_y(v)), a)))
        a = np.array([np.full_like(e[i], a[i]) for i in range(len(a))]).flatten()
        e = e.flatten()
        x2_max = x[:,1].max()
        df = pd.DataFrame()
        df['x1'] = x[:, 0]
        df['x2'] = x[:, 1]
        df['category'] = y
        fg = sns.lmplot(x='x1', y='x2', hue='category', data=df, fit_reg=False)
        #fg.axes[0, 0].plot(a[e<x2_max], e[e<x2_max], )
        plt.show()
