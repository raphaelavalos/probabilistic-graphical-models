import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class LDA:
    def __init__(self):
        self.sigma = np.identity(2)
        self.inv_sigma = np.identity(2)
        self.pi = 0
        self.mu0 = np.zeros((2,))
        self.mu1 = np.zeros((2,))
        self.w = np.zeros((2,))
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
        self.w = - self.inv_sigma @ (self.mu1 - self.mu0)
        self.alpha = ((1 - self.pi) / self.pi) * np.exp(
            .5 * ((self.mu0.T @ self.inv_sigma @ self.mu0) - (self.mu1.T @ self.inv_sigma @ self.mu1)))
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise NameError('Not fitted')
        tmp = np.log(self.alpha) - x @ self.w
        predictions = np.zeros(len(x))
        predictions[tmp > 0] = 1
        return predictions

    def score(self, x, y):
        return np.count_nonzero(self.predict(x) == y) / len(y)

    def plot(self, x, y):
        if not self.fitted:
            raise NameError('Not fitted')
        a = np.array([x[:, 0].min(), x[:, 0].max()])
        sep = lambda e: (np.log(self.alpha) - self.w[0] * e) / self.w[1]
        df = pd.DataFrame()
        df['x1'] = x[:, 0]
        df['x2'] = x[:, 1]
        df['category'] = y
        fg = sns.lmplot(x='x1', y='x2', hue='category', data=df, fit_reg=False)
        fg.axes[0, 0].plot(a, sep(a), c="black")
        plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def inv_sigmoid(x):
    return sigmoid(x) * sigmoid(-x)


class IRLS:
    def __init__(self, max_step=10000, alpha=.4, beta=.8, modified=False, epsilon=10 ** (-4)):
        self.w = np.random.rand(3)
        self.epsilon = epsilon
        self.max_step = max_step
        self.alpha = alpha
        self.beta = beta
        self.modifed = modified
        self.fitted = False

    # In the following code the hessian and the grad is not of - log likelihood

    def normal_fit(self, x, y):
        X = np.c_[x, np.ones(len(x))]
        w = self.w
        step = 0
        lambda_square = self.epsilon + 1

        def log_likelihood(w):
            return np.log(sigmoid((2 * y - 1) * (X @ w))).sum()

        while (step < self.max_step) and (lambda_square > self.epsilon):
            eta = sigmoid(X @ w)
            hessian = X.T @ np.diagflat(eta * (1 - eta)) @ X
            hessian_inv = np.linalg.inv(hessian)
            grad = - X.T @ (y - eta)
            delta_w = - hessian_inv @ grad
            lambda_square = - grad.T @ delta_w
            t = 1
            lw = log_likelihood(w)
            limit = self.alpha * grad @ delta_w
            while lw - log_likelihood(w + t * delta_w) >= t * limit:
                t *= self.beta
            w += t * delta_w
            step += 1
        self.w = w
        self.fitted = True

    def modified_fit(self, x, y):
        print('modified fit')
        X = np.c_[x, np.ones(len(x))]
        w = self.w
        update = [self.epsilon + 1]
        step = 0
        while (step < self.max_step) and (np.linalg.norm(update, 2) > self.epsilon):
            eta = sigmoid(X @ w)
            update = np.linalg.inv(X.T @ np.diagflat(eta) @ X) @ X.T @ (y - eta)
            w += update
            step += 1
        self.w = w
        self.fitted = True

    def fit(self, x, y):
        if self.modifed:
            self.modified_fit(x, y)
        else:
            self.normal_fit(x, y)

    def predict(self, x, w=None):
        if not self.fitted and (w is None):
            raise NameError('Not fitted')
        if w is None:
            w = self.w
        X = np.c_[x, np.ones(len(x))]
        proba = sigmoid(X @ w)
        prediction = np.zeros(len(x))
        prediction[proba > .5] = 1
        return prediction

    def score(self, x, y, w=None):
        return np.count_nonzero(self.predict(x, w) == y) / len(y)

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
        fg.axes[0, 0].plot(a, sep(a), c="black")
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
        fg.axes[0, 0].plot(a, sep(a), c="black")
        plt.show()


class QDA:
    def __init__(self):
        self.sigma0 = np.identity(2)
        self.sigma1 = np.identity(2)
        self.inv_sigma0 = np.identity(2)
        self.inv_sigma1 = np.identity(2)
        self.pi = 0
        self.mu0 = np.zeros((2,))
        self.mu1 = np.zeros((2,))
        self.w = np.zeros(2, )
        self.alpha = 0
        self.a = 0
        self.P = np.zeros((2, 2))
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
        self.a = np.log(
            ((1 - self.pi) / self.pi) * np.sqrt(np.linalg.det(self.sigma0) / np.linalg.det(self.sigma1))) + .5 * (
                         self.mu0.T @ self.inv_sigma0 @ self.mu0 - self.mu1.T @ self.inv_sigma1 @ self.mu1)
        self.w = self.inv_sigma1 @ self.mu1 - self.inv_sigma0 @ self.mu0
        self.P = .5 * (self.inv_sigma0 - self.inv_sigma1)
        self.poly_on_y = lambda e: np.array(
            [self.P[1, 1], 2 * self.P[1, 0] * e + self.w[1], self.a + e * self.w[0] + self.P[0, 0] * e * e])
        self.fitted = True

    def predict(self, x):
        if not self.fitted:
            raise NameError('Not fitted')
        probas = self.a + x @ self.w + np.einsum('ij,ij -> i', x @ self.P, x)
        predictions = np.zeros(len(x))
        predictions[probas > 0] = 1
        return predictions

    def score(self, x, y):
        return np.count_nonzero(self.predict(x) == y) / len(y)

    def plot(self, x, y):
        if not self.fitted:
            raise NameError('Not fitted')
        a = np.linspace(x[:, 0].min(), x[:, 0].max(), 1000)
        e = np.array(list(map(lambda v: np.roots(self.poly_on_y(v)), a)))
        k, l = np.array(list(
            zip(*[(np.full_like(np.real(e[i]), a[i]), np.real(e[i])) for i in range(len(a)) if np.isreal(e[i]).any()])))
        k = k.flatten()
        l = l.flatten()
        df = pd.DataFrame()
        df['x1'] = x[:, 0]
        df['x2'] = x[:, 1]
        df['category'] = y
        fg = sns.lmplot(x='x1', y='x2', hue='category', data=df, fit_reg=False)
        fg.axes[0, 0].scatter(k, l, s=1, c='black')
        plt.show()
