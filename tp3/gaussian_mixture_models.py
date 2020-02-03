import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import mixture
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

from tp3.algorithm_singledim_em import singleDimEm
from tp3.algorithm_em import myEM

import pandas as pd

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

data = np.concatenate([s1, s2], axis=0)
y = np.concatenate([y1, y2])

em = singleDimEm(n_components=2, dim=1)
em.fit(data, nb_iteration=1)

print("end")

# ======================
# Simulated data 2D
# ======================

# =========================
# Simulated Datasets
# =========================

# First simulated data set
X1, Y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)


# Second simulated data set
X2, Y2 = make_blobs(n_samples=200, n_features=2, centers=3)


# Third simulated data set
X3, Y3 = make_moons(n_samples=200, shuffle=True, noise=None, random_state=None)




# =========================
# Breast Cancer
# =========================

bc_data = pd.read_table("data/wdbc.data", sep=",", header=None)

X_bc = bc_data[bc_data.columns[2:]]
Y_bc = bc_data[bc_data.columns[1]]
Y_bc = Y_bc.astype("category").cat.rename_categories(range(0, Y_bc.nunique()))

X_bc = X_bc.values
Y_bc = Y_bc.values

# =========================
# Mice
# =========================

def sanitize(x):
    if x is None:
        return 0.
    elif isinstance(x, str):
        return float(x.replace(",", "."))
    else:
        return x


mice_data = pd.read_table("data/Data_Cortex_Nuclear.csv", sep=",")

X_mice = mice_data.drop(["MouseID", "class"], axis=1)[mice_data.columns[1:78]]
X_mice = X_mice.applymap(sanitize)
X_mice = X_mice.fillna(0)

Y_mice = mice_data["class"]
Y_mice = Y_mice.astype("category").cat.rename_categories(range(0, Y_mice.nunique()))

X_mice = X_mice.values
Y_mice = Y_mice.values

# Combined dataset

distrib_names = ["Two informative features, one cluster per class",
                 "Three blobs",
                 "Non-linearly separated data sets",
                 "Mice Protein Expression",
                 "Breast Cancer"]

datasets_X = [X1, X2, X3, X_mice, X_bc]
datasets_Y = [Y1, Y2, Y3, Y_mice, Y_bc]

# Second simulated data set

number_clusters = [2,3,2, 8, 2]


for i, (X, Y, k, name) in enumerate(zip(datasets_X, datasets_Y, number_clusters, distrib_names)):
    print(name)
    if i <= len(distrib_names) - 3:
        em = myEM(n_components=k, dim=2)
        em.fit(X, 10)

        plt.title(f"{name}", fontsize='small')
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=Y, s=25, edgecolor='k')
        plt.scatter(em.mu[:, 0], em.mu[:,1], marker='X', c="red", s=100, edgecolors='k')

        plt.savefig(f"exports/homemade-EM-{name}.png")
        plt.clf()


    lowest_bic = np.infty
    best_gmm = None
    best_n_component = 0
    bic = []
    n_components_range = range(1, 5)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))

            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm
                best_n_component = n_components

    print(f"{name} -- {lowest_bic=}, {best_n_component=}")
    y_predicted = best_gmm.predict(X)

    if i <= len(distrib_names) - 3:
        plt.scatter(X[:, 0], X[:, 1], marker='o', c=y_predicted, s=25, edgecolor='k')
        plt.savefig(f"exports/EM-{name}.png")
        plt.clf()













