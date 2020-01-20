import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons
from sklearn import linear_model, datasets

from sklearn.model_selection import cross_val_score

from mllab import *
from logistic_regression import LogisticRegression


# ==================================
# Datasets
# ==================================

# First simulated data set
# plt.title("Two informative features, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)
# plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')

# Second simulated data set
# plt.title("Three blobs", fontsize='small')
X2, Y2 = make_blobs(n_samples=200, n_features=2, centers=3)
# plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2, s=25, edgecolor='k')

# Third simulated data set
#plt.title("Non-linearly separated data sets", fontsize='small')
X3, Y3 = make_moons(n_samples=200, shuffle=True, noise=None, random_state=None)

# plt.scatter(X3[:, 0], X3[:, 1], marker='o', c=Y3, s=25, edgecolor='k')

# =========================
# Breast Cancer
# =========================

bc_data = pd.read_table("data/wdbc.data", sep=",", header=None)

X_bc = bc_data[bc_data.columns[2:]]
Y_bc = bc_data[bc_data.columns[1]]
Y_bc = Y_bc.astype("category").cat.rename_categories(range(0, Y_bc.nunique())).astype(int)

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

datasets_name = ["Two informative features, one cluster per class", "Three blobs", "Non-linearly separated data sets",
                 "Breast Cancer"]
X_datasets = [X1, X2, X3, X_bc]
Y_datasets = [Y1, Y2, Y3, Y_bc]


# ==================================
# Datasets
# ==================================

for X, Y, name in zip(X_datasets, Y_datasets, datasets_name):

    # Sklearn logistic regression
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, Y)

    if X.shape[1] == 2:
        map_regions(logreg, X, Y=Y, num=1000)
        plt.savefig(f"exports/{name}-regions_logreg.png")

    # Cross validation evaluation
    logreg = linear_model.LogisticRegression(C=1e5)
    scores = cross_val_score(logreg,X=X, y=Y, cv=10)

    print(f"{name}: LogReg cross validation score (accuracy): {np.mean(scores)}")

    clf = LogisticRegression(dim=X.shape[1])

    clf.fit(X, Y)
    if X.shape[1] == 2:
        map_regions(clf, X, Y=Y, num=1000)
        plt.savefig(f"exports/{name}-regions_mylogreg.png")

    # Cross validation evaluation
    clf = LogisticRegression(dim=X.shape[1])
    scores = cross_val_score(clf,X=X, y=Y, cv=10)

    print(f"{name}: My LogReg cross validation score (accuracy): {np.mean(scores)}")


