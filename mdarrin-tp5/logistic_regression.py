

import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegression():
    def __init__(self, dim=2):
        self.dim = dim
        self.theta = np.zeros(dim)

    def p(self, x):
        exp = np.exp(np.matmul(x, self.theta))

        p = np.array([1/(1+exp), exp / (1+exp)]).transpose()

        return p

    def get_params(self, *args, **kwargs):
        return {"dim": self.dim}

    def optimization_step(self, X, Y):
        dl = - np.sum(X * (Y - self.p(X)[:, 1]).reshape(-1, 1), axis=0)
        p = self.p(X)
        dl2 = sum(np.dot(X[i], X[i]) * p[i, 1] * (1 - p[i, 1]) for i in range(X.shape[0]))

        self.theta -= (1/dl2) * dl

    def _loss(self):
        pass

    def decision_function(self, X):
        return np.matmul(X, self.theta)

    def fit(self, X, Y, steps=10):
        for i in range(steps):
            self.optimization_step(X, Y)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        return self.p(X)

    def fit_predict(self, X, Y, steps=10):
        self.fit(X, Y, steps)
        return self.predict(X)

    def score(self, X, Y):
        return accuracy_score(self.predict(X), Y)