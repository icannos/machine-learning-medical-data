import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_moons

import pandas as pd

from collections import defaultdict
import numpy as np

def computes_metrics(metrics_fn, X, Y, Ychap):
    return pd.Series([metric(Y, Ychap) if metric != metrics.silhouette_score else metric(X, Ychap)
               for metric in metrics_fn])


# =========================
# Simulated Datasets
# =========================

# First simulated data set
plt.title("Two informative features, one cluster per class", fontsize='small')
X1, Y1 = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)
plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1, s=25, edgecolor='k')
plt.savefig("exports/twoinformativedataset.png")
plt.clf()

# Second simulated data set
plt.title("Three blobs", fontsize='small')
X2, Y2 = make_blobs(n_samples=200, n_features=2, centers=3)
plt.scatter(X2[:, 0], X2[:, 1], marker='o', c=Y2, s=25, edgecolor='k')
plt.savefig("exports/trhee_blobs.png")
plt.clf()

# Third simulated data set
plt.title("Non-linearly separated data sets", fontsize='small')
X3, Y3 = make_moons(n_samples=200, shuffle=True, noise=None, random_state=None)
plt.scatter(X3[:, 0], X3[:, 1], marker='o', c=Y3, s=25, edgecolor='k')
plt.savefig("exports/nonlin_dataset.png")
plt.clf()



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
number_clusters = [2,3,3, 8, 2]


metrics_names = ["homogeneity_score", "completeness_score", "v_measure_score",
                 "adjusted_rand_score", "silhouette_score"]

metrics_fn = [metrics.homogeneity_score,
              metrics.completeness_score,
              metrics.v_measure_score,
              metrics.adjusted_rand_score,
              metrics.silhouette_score
              ]


metrics_df = defaultdict(dict)

output_str = ""

# On étudie chacun des datasets
for i, (X, Y, k) in enumerate(zip(datasets_X, datasets_Y, number_clusters)):
    print(f'{distrib_names[i]}')
    plt.subplots_adjust(hspace=0.6)
    plt.figure(1, figsize=(50,50), dpi=100)
    plt.title(f"{distrib_names[i]}")
    # =========================
    # Clustering for each dataset
    # =========================

    plt.subplot(3, 2, 1)

    plt.scatter(X[:, 0], X[:, 1], s=10, c=Y)
    plt.title("True labels")

    # Kmeans
    km = KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
    km.fit(X)

    plt.subplot(3, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=km.labels_)
    plt.title(f"Kmean")

    # Hiearachical Clustering

    m_hclustring = {}
    t = 1
    for linkage in ('ward', 'average', 'complete'):
        plt.subplot(3, 2, 2+t)
        t+=1
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=k)
        clustering.fit(X)
        plt.scatter(X[:, 0], X[:, 1], s=10, c=clustering.labels_)
        plt.title(f"Hclustering {linkage}")
        m_hclustring[linkage] = computes_metrics(metrics_fn, X, Y, clustering.labels_)

    # Spectral clustering
    plt.subplot(3, 2, 6)
    spectral = cluster.SpectralClustering(n_clusters=k, eigen_solver='arpack', affinity="nearest_neighbors")
    spectral.fit(X)
    plt.scatter(X[:, 0], X[:, 1], s=10, c=spectral.labels_)
    plt.title(f"Spectral")

    plt.savefig(f'exports/{distrib_names[i]}-clustering.png')
    plt.clf()

    # =========================
    # Metrics for each dataset
    # =========================

    m_spectral = computes_metrics(metrics_fn, X, Y, spectral.labels_)
    m_kmean = computes_metrics(metrics_fn, X, Y, km.labels_)

    metrics_df[distrib_names[i]]["kmean"] = m_kmean
    metrics_df[distrib_names[i]]["spectral"] = m_spectral
    for key,v in m_hclustring.items():
        metrics_df[distrib_names[i]][f"hierarchical-{key}"] = v

    mean_scores = [m_kmean.mean(), m_spectral.mean()] + [v.mean() for key,v in m_hclustring.items()]
    algos = ["kmean", "spectral"] + ["hiearchical-" + key for key,v in m_hclustring.items()]
    output_str += f'Best algorithm for {distrib_names[i]} is {algos[np.argmax(mean_scores)]} ' \
                  f'with mean scores {np.max(mean_scores)} \n'

for k,v in metrics_df.items():
    v["metrics"] = pd.Series(metrics_names)

    df = pd.DataFrame(v).round(2)
    df = df.set_index("metrics")

    df.to_csv(f"exports/{k}-scores.csv")

print(output_str)

