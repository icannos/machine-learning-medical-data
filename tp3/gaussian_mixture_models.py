import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import mixture
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

from tp3.algorithm_em import singleDimEM

# ======================
# Simulated data: one dimension case
# ======================

mu1, sigma1 = 0, 0.3  # mean and standard deviation
s1 = np.random.normal(mu1, sigma1, 100)
y1 = np.repeat(0, 100)

mu2, sigma2 = 2, 0.3  # mean and standard deviation
s2 = np.random.normal(mu2, sigma2, 100)
y2 = np.repeat(1, 100)

mu = [mu1, mu2]
sigma = [sigma1, sigma2]

data = np.concatenate([s1, s2])
y = np.concatenate([y1, y2])

em = singleDimEM(n_components=2)
em.fit(data)



# ======================
# Simulated data
# ======================











