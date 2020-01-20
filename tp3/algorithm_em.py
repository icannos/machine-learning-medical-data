import numpy as np
import random
from scipy.stats import multivariate_normal, norm


def pr_single_comp(mu, sigma, x):
    if len(sigma.shape[1:]) == 2:
        return None
        return multivariate_normal(mean=mu, cov=sigma).pdf(x)
    else:
        proba = []

        for i in range(sigma.shape[0]):
            proba.append(np.transpose(norm(loc=mu[i], scale=sigma[i]).pdf(x)))

        proba = np.array(proba)
        return np.transpose(proba)

def pr_single_normalized(mu, sigma, x):
    unnorm_prob = pr_single_comp(mu, sigma, x)
    normalization = np.sum(unnorm_prob, axis=1)

    return unnorm_prob / normalization.reshape((-1, 1))


def update_mu(x, mu, sigma):
    prob = pr_single_normalized(mu, sigma, x)
    print(prob.shape)
    print(x.shape)
    hat_mu = np.zeros(mu.shape)
    for i in range(0, prob.shape[1]):
        hat_mu  += prob[:,i] * x
    hat_mu = hat_mu / np.sum(pr_single_normalized(mu, sigma, x), axis=1).reshape((-1, 1))
    return hat_mu


def update_sigma(x, mu, sigma):
    prob = pr_single_normalized(mu, sigma, x)

    hat_sigma = np.zeros(sigma.shape)
    for i in range(0, prob.shape[1]):
        hat_sigma += prob[:, i] * (x - mu) ** 2
    hat_sigma = hat_sigma / np.sum(pr_single_normalized(mu, sigma, x), axis=1).reshape((-1, 1))
    return hat_sigma


class singleDimEM:
    def __init__(self, n_components=1, dim=1):
        self.dim = dim
        self.n_components = n_components
        self.mu = None
        self.sigma = None

        self.reset()

    def reset(self):
        self.mu = np.random.uniform(-5, 5, size =(self.n_components, self.dim))

        if self.dim > 1:
            self.sigma = np.random.uniform(0, 1, size =(self.n_components, self.dim, self.dim))
        else:
            self.sigma = np.random.uniform(0, 1, size=(self.n_components, self.dim))


    def fit(self, data, nb_iteration = 10):
        # Learning procedure (optimization)

        for iter in range(1, nb_iteration):
            hat_mu = update_mu(data, self.mu, self.sigma)
            hat_sigma = update_sigma(data, self.mu, self.sigma)
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
           return np.array([multivariate_normal(mean=self.mu[i], cov = self.sigma[i]).pdf(X)
                            for i in range(self.n_components)])
