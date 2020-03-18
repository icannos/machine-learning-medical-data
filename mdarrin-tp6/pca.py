import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score

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
# Golub data
# =========================

X_golub = pd.read_csv("data/Golub_X", sep=" ").values  # Observations
Y_golub = pd.read_csv("data/Golub_y", sep=" ").values
Y_golub = np.reshape(Y_golub, Y_golub.shape[:-1])

# Combined dataset

distrib_names = ["Two informative features, one cluster per class",
                 "Three blobs",
                 "Non-linearly separated data sets",
                 "Mice Protein Expression",
                 "Breast Cancer"]

datasets_name = ["Golub", "Breast Cancer"]
n_clusters = [2,2]
X_datasets = [X_golub, X_bc]
Y_datasets = [Y_golub, Y_bc]

for X, Y, name, k in zip(X_datasets, Y_datasets, datasets_name, n_clusters):
    # =========================
    # Plotting PCA
    # =========================

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)
    plt.title(f"{name}: PCA with true clusters")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Y, s=25, edgecolor='k')

    km = KMeans(n_clusters=k)
    km.fit(X)
    Ychap = km.predict(X)

    plt.subplot(2, 1, 2)
    plt.title(f"{name}: PCA with kmean clusters")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Ychap, s=25, edgecolor='k')

    plt.savefig(f"exports/{name}-naivepcaclustering.png")
    plt.clf()

    # =========================
    # Kernal PCA
    # =========================

    pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10, n_components=2)

    pca.fit(X)
    X_pca = pca.transform(X)

    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)
    plt.title(f"{name}: KernelPCA with true clusters")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Y, s=25, edgecolor='k')

    km = KMeans(n_clusters=k)
    km.fit(X)
    Ychap = km.predict(X)

    plt.subplot(2, 1, 2)
    plt.title(f"{name}: KernelPCA with kmean clusters")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Ychap, s=25, edgecolor='k')

    plt.savefig(f"exports/{name}-kernelpcaclustering.png")
    plt.clf()

    # =========================
    # Incremental PCA
    # =========================

    pca = IncrementalPCA(n_components=2, batch_size=10)

    pca.fit(X)
    X_pca = pca.transform(X)

    plt.subplots_adjust(hspace=0.4)
    plt.subplot(2, 1, 1)
    plt.title(f"{name}: Incremental PCA with true clusters")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Y, s=25, edgecolor='k')

    km = KMeans(n_clusters=k)
    km.fit(X)
    Ychap = km.predict(X)

    plt.subplot(2, 1, 2)
    plt.title(f"{name}: Incremental PCA with kmean clusters")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=Ychap, s=25, edgecolor='k')

    plt.savefig(f"exports/{name}-increcaclustering.png")
    plt.clf()

    # =========================
    # Classification with PCA
    # =========================

    x_compo = list(range(2, 31, 1))
    y_mean = []
    y_var = []
    for c in x_compo:
        pca = PCA(n_components=c)
        pca.fit(X)
        X_pca = pca.transform(X)

        clf = LogisticRegression()

        scores = cross_val_score(clf, X_pca, Y, cv=10)
        y_mean.append(np.mean(scores))
        y_var.append(np.var(scores))

    errors = 2*np.sqrt(y_var)

    plt.subplot(2, 1, 1)
    plt.title(f"{name}: PCA with Logistic Regression")
    plt.plot(x_compo, y_mean, 'k', color='#CC4F1B')
    plt.fill_between(x_compo, y_mean - errors, y_mean + errors, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

    x_compo = list(range(2, 31, 1))
    y_mean = []
    y_var = []
    for c in x_compo:
        pca = PCA(n_components=c)
        pca.fit(X)
        X_pca = pca.transform(X)

        clf = svm.SVC(kernel='rbf')

        scores = cross_val_score(clf, X_pca, Y, cv=10)
        y_mean.append(np.mean(scores))
        y_var.append(np.var(scores))

    errors = 2*np.sqrt(y_var)

    plt.subplot(2, 1, 2)
    plt.title(f"{name}: PCA with SVM")
    plt.plot(x_compo, y_mean, 'k', color='#CC4F1B')
    plt.fill_between(x_compo, y_mean - errors, y_mean + errors, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    plt.savefig(f"exports/{name}-pca-classification.png")


