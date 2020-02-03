import numpy as np
import random
from scipy.stats import multivariate_normal, norm


class myEM:
    def __init__(self, n_components=1, dim=1):
        self.dim = dim
        self.n_components = n_components
        self.mu = None
        self.sigma = None

        self.reset()

    def reset(self):
        self.sigma = np.random.uniform(-1, 1, size=(self.n_components, self.dim, self.dim))
        for i in range(self.n_components):
            self.sigma[i] = np.matmul(self.sigma[i], np.transpose(self.sigma[i]))

        self.mu = np.random.uniform(-3, 3, size=(self.n_components, self.dim))

    def fit(self, data, nb_iteration=100):
        # Learning procedure (optimization)

        for iter in range(1, nb_iteration):
            hat_mu = self.update_mu(data)
            hat_sigma = self.update_sigma(data)

            self.mu = hat_mu
            self.sigma = hat_sigma + 1e-13
    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        y = []
        for i in range(X.shape[0]):
            y.append([multivariate_normal(mean=self.mu[j], cov=self.sigma[j]).pdf(X[i])
                      for j in range(self.n_components)])

        return np.array(y)

    def update_mu(self, X):
        pnk = self.proba_nk(X)

        mu = np.zeros((self.n_components, *X.shape[1:]))

        for k in range(self.n_components):
            mu[k] = np.sum(pnk[:, k].reshape(-1,1)*X, axis=0) / (np.sum(pnk[:, k]).reshape(-1,1)+1E-10)

        return mu

    def update_sigma(self, X):
        sigma = np.zeros((self.n_components, self.dim, self.dim))
        pnk = self.proba_nk(X)

        for k in range(self.n_components):
            sigma[k] = np.cov(np.transpose(X), aweights=pnk[:, k]+1E-10)

        return sigma

    def proba_x(self, X):
        probs = self.predict_proba(X)
        probk = self.proba_k(X)

        p = np.zeros(X.shape[0])

        for k in range(self.n_components):
            p += probs[:, k] * probk[k]

        return p

    def proba_nk(self, X):
        px = self.proba_x(X)
        pk = self.proba_k(X)
        p = self.predict_proba(X)

        p = p * pk
        pnk =  p / px.reshape((-1,1))

        return pnk

    def proba_k(self, X):
        probs = self.predict_proba(X)
        normalization = np.sum(probs, axis=0)

        return normalization / np.sum(normalization)
