import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectFdr
from sklearn import linear_model

# =========================
# Golub data
# =========================

X_golub = pd.read_csv("data/Golub_X", sep=" ").values  # Observations
Y_golub = pd.read_csv("data/Golub_y", sep=" ").values
Y_golub = np.reshape(Y_golub, Y_golub.shape[:-1])

X_breast = pd.read_csv("../tp6/data/Breast.txt", sep=' ')
Y_breast = X_breast.as_matrix()[:, 30]
X_breast = X_breast.as_matrix()[:, 0:29]

datasets_names = ["golub", "breast"]
X_datasets = [X_golub, X_breast]
Y_datasets = [Y_golub, Y_breast]

for X, Y, name in zip(X_datasets, Y_datasets, datasets_names):
    # =========================
    # Variance Threshold
    # =========================

    var = np.var(X, axis=0)
    print(
        f"{name}: Number of features {X.shape[1]}, Max Variance {np.max(var)}, Mean Variance {np.mean(var)}, Min Variance {np.min(var)}")

    t_1 = 0.01
    t_2 = 0.05

    vt_1 = VarianceThreshold(threshold=t_1)
    vt_2 = VarianceThreshold(threshold=t_2)
    X_1 = vt_1.fit_transform(X)
    X_2 = vt_2.fit_transform(X)

    print(f'{name}: Variance threshold={t_1}, Number of features: {X_1.shape[1]}')
    print(f'{name}: Variance threshold={t_2}, Number of features: {X_2.shape[1]}')

    # =========================
    # Univariate selection with stat test
    # =========================

    fdr = SelectFdr()
    X_fdr = fdr.fit_transform(X, Y)

    print(f'{name}: FDR, Number of features: {X_fdr.shape[1]}')

    # =========================
    # L1 Based
    # =========================

    # Linear Lasso
    # Attention ici le sujet donne un algo de régression linéaire pas de régression logistique !
    alphas = np.linspace(0.01, 1, 1000)
    scores = []
    features_number = []
    for alpha in alphas:
        clf = linear_model.LogisticRegression(penalty='l1', C=alpha, solver='liblinear')
        clf.fit(X, Y)

        scores.append(clf.score(X,Y))
        features_number.append(np.sum(np.abs(clf.coef_) > 0.05))

    plt.figure(1, figsize=(30, 10), dpi=100)

    plt.subplot(2, 3, 1)
    plt.plot(alphas, scores)
    plt.title(f"{name}: Logistic Regression")
    plt.ylabel("Accuracy")
    #plt.xlabel("Penalty coefficient: alpha")

    plt.subplot(2, 3, 4)
    plt.plot(alphas, features_number)
    plt.ylabel("Number of features")
    plt.xlabel("Penalty coefficient: C")

    # plt.savefig(f"exports/{name}-lassoreg.png")
    # plt.clf()


    # ===========================================================

    # LinearSVC
    alphas = np.linspace(0.01, 1, 100)
    scores = []
    features_number = []
    for alpha in alphas:

        clf = LinearSVC(penalty='l1', C=alpha, dual=False)
        clf.fit(X, Y)

        scores.append(clf.score(X,Y))
        features_number.append(np.sum(np.abs(clf.coef_) > 0.05))

    plt.subplot(2, 3, 2)
    plt.title(f"{name}: Linear SVC")
    plt.plot(alphas, scores)
    plt.ylabel("Accuracy")
    #plt.xlabel("Penalty coeficient")

    plt.subplot(2, 3, 5)
    plt.plot(alphas, features_number)
    plt.ylabel("Number of features")
    plt.xlabel("Penalty coeficient: C")

    #plt.savefig(f"exports/{name}-linearSVC.png")
    #plt.clf()

    # ===========================================================

    # Elastic Net
    alphas = np.linspace(0.01, 1, 100)
    scores = []
    features_number = []
    for alpha in alphas:

        clf = linear_model.ElasticNet(alpha=alpha, l1_ratio=0.7)
        clf.fit(X, Y)

        scores.append(clf.score(X,Y))
        features_number.append(np.sum(np.abs(clf.coef_) > 0.05))

    plt.subplot(2, 3, 3)
    plt.title(f"{name}: ElasticNet")
    plt.plot(alphas, scores)
    plt.ylabel("Accuracy")
    #plt.xlabel("Penalty coeficient: alpha")

    plt.subplot(2, 3, 6)
    plt.plot(alphas, features_number)
    plt.ylabel("Number of features")
    plt.xlabel("Penalty coeficient: alpha")

    #plt.savefig(f"exports/{name}-elasticnet.png")
    plt.savefig(f"exports/{name}.png")
    plt.clf()




