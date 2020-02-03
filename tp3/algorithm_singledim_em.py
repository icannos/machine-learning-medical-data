import numpy as np
import random
from scipy.stats import multivariate_normal, norm


def pr_single_comp(mu, sigma, x):
    proba = []

    for i in range(len(sigma)):
        proba.append(np.transpose(norm(loc=mu[i], scale=sigma[i]).pdf(x)))

    proba = np.array(proba)
    return np.transpose(proba)


def pr_single_normalized(mu, sigma, x):
    unnorm_prob = pr_single_comp(mu, sigma, x)
    normalization = np.sum(pr_single_comp(mu, sigma, x), axis=1)
    prob = []
    for i in range(0, len(unnorm_prob)):
        prob.append(unnorm_prob[:][i] / normalization[i])
    return prob


def update_mu(x, mu, sigma):
    prob = pr_single_normalized(mu, sigma, x)
    hat_mu = [0 for _ in range(len(mu))]
    for i in range(0, len(prob)):
        hat_mu = hat_mu + prob[i][:] * x[i,]
    hat_mu = hat_mu / np.sum(pr_single_normalized(mu, sigma, x), axis=0)
    return hat_mu


def update_sigma(x, mu, sigma):
    prob = pr_single_normalized(mu, sigma, x)
    hat_sigma = [0 for _ in range(len(mu))]
    for i in range(0, len(prob)):
        hat_sigma = hat_sigma + prob[i][:] * (x[i,] - mu) ** 2
    hat_sigma = hat_sigma / np.sum(pr_single_normalized(mu, sigma, x), axis=0)
    return hat_sigma


class singleDimEm:
    def __init__(self, n_components=1, dim=1):
        self.dim = dim
        self.n_components = n_components
        self.mu = None
        self.sigma = None

        self.reset()

    def reset(self):
            # self.sigma = np.random.uniform(0, 1, size=(self.n_components))
            self.mu = [np.random.uniform(-2, 2), np.random.uniform(0, 4)]
            self.sigma = [0.3, 0.3]

    def fit(self, data, nb_iteration=100):
        # Learning procedure (optimization)

        for iter in range(1, nb_iteration):
            hat_mu = update_mu(data, self.mu, self.sigma)
            hat_sigma = update_sigma(data, hat_mu, self.sigma)
            print('iter', iter)
            print("updated mu = ", hat_mu)
            print("updated sigma = ", hat_sigma)
            self.mu = hat_mu
            self.sigma = hat_sigma + 1e-13

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return np.array([multivariate_normal(mean=self.mu[i], cov=self.sigma[i]).pdf(X)
                         for i in range(self.n_components)])
