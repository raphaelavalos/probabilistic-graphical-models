import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse


class HMM:
    def __init__(self, A, K, pi, mus, sigmas, data):
        self.A = A
        self.K = K
        self.pi = pi
        self.mus = mus
        self.sigmas = sigmas
        self.data = data
        self.T = data.shape[0]
        self.log_alphas = np.zeros((self.T, K))
        self.log_betas = np.zeros((self.T, K))
        self.log_emmission = np.zeros((self.T, K))
        self.e1 = np.zeros((self.T, self.K))
        self.e2 = np.zeros((self.T, self.K, self.K))

    def compute_alpha(self):
        self.log_alphas[0] = np.log(self.pi) + self.log_emmission[0]
        for t in range(1, self.T):
            tmp = (np.log(self.A) + self.log_alphas[t - 1]).max(axis=1)
            self.log_alphas[t] = self.log_emmission[t] + tmp + np.log(
                np.exp(np.log(self.A) + self.log_alphas[t - 1] - tmp[:, None]).sum(axis=1))

    def compute_beta(self):
        self.log_betas[- 1] = 0
        for t in range(self.T - 2, -1, -1):
            tmp = (self.log_emmission[t + 1] + np.log(self.A.T) + self.log_betas[t + 1]).max(axis=1)
            self.log_betas[t] = tmp + np.log(
                np.exp(self.log_emmission[t + 1] + np.log(self.A.T) + self.log_betas[t + 1] - tmp[:, None]).sum(axis=1))

    def compute_log_emmision(self):
        for k in range(self.K):
            self.log_emmission[:, k] = multivariate_normal(self.mus[k], self.sigmas[k]).logpdf(self.data)

    def compute_probas(self):
        self.compute_log_emmision()
        self.compute_alpha()
        self.compute_beta()

    def estimate(self):
        a = self.log_alphas + self.log_betas
        a_star = a.max(axis=1)
        self.e1 = np.exp(a - a_star[:, None] - np.log(np.exp(a - a_star[:, None]).sum(axis=1)[:, None]))
        self.e2 = np.exp(
            self.log_alphas[:-1, None] + self.log_betas[1:, :, None] + np.log(self.A) +
            self.log_emmission[1:, :, None] - a_star[-1] -
            np.log(np.exp(a[-1] - a_star[-1]).sum()))

    def maximisation(self):
        self.pi = self.e1[0, :]
        self.A = self.e2.sum(0) / self.e2.sum(0).sum(0)[:, None]
        mus = self.e1.T @ self.data / self.e1.sum(axis=0)[:, None]
        tmp = self.data[:, None] - self.mus[None,]
        self.sigmas = (self.e1[:, :, None] * tmp).swapaxes(0, 1).swapaxes(1, 2) @ tmp.swapaxes(0, 1) / self.e1.sum(
            axis=0)[:, None, None]
        self.mus = mus

    def em(self, n):
        for _ in range(n):
            self.compute_probas()
            self.estimate()
            self.maximisation()

    def viterbi(self, data=None, plot=True):
        if data is None:
            data = self.data
        T1 = np.zeros((self.T, self.K))
        T2 = np.zeros((self.T, self.K))
        log_emmission = np.zeros((self.T, self.K))
        labels = np.zeros((self.T), dtype=np.int8)
        for k in range(self.K):
            log_emmission[:, k] = multivariate_normal(self.mus[k], self.sigmas[k]).logpdf(data)
        T1[0] = np.log(self.pi) + log_emmission[0]
        for t in range(1, self.T):
            for j in range(self.K):
                T1[t, j] = (T1[t - 1] + np.log(self.A[j]) + log_emmission[t, j]).max()
                T2[t, j] = (T1[t - 1] + np.log(self.A[j]) + log_emmission[t, j]).argmax()
        labels[-1] = T1[-1].argmax()
        for t in range(self.T - 2, -1, -1):
            labels[t] = T2[t + 1, labels[t + 1]]
        if plot:
            self.plot(data, labels)
        log_likelihood = np.log(self.pi[labels[0]]) + np.log(self.A[labels[1:], labels[:-1]]).sum() + \
                         log_emmission[range(500), labels].sum()
        print('Log-likelihood: %.2f' % log_likelihood)
        return labels

    def plot(self, data=None, labels=None):
        if data is None:
            data = self.data
        s = 4.605
        palette = sns.color_palette("husl", n_colors=self.K)
        ax = sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette=palette)
        plt.scatter(x=self.mus[:, 0], y=self.mus[:, 1], color='black')

        for k in range(self.K):
            l, P = np.linalg.eigh(self.sigmas[k])
            ax.add_artist(Ellipse(self.mus[k], width=2 * np.sqrt(l[0] * s), height=2 * np.sqrt(l[1] * s),
                                  angle=np.arctan(P[1, 0] / P[0, 0]) * 180 / np.pi, facecolor=None, fill=False,
                                  color='black'))
        plt.show()
