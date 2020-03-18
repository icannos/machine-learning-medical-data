import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer, Activation

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, make_scorer

from sklearn.neural_network import MLPClassifier


# =========================
# SPLEX
# =========================

classes = pd.read_table("data/classes.txt", sep=" ", header=0)

splex_host = pd.read_table("data/SPLEX_host.txt", sep=" ", header=0, dtype=float).values
splex_env = pd.read_table("data/SPLEX_env.txt", sep=" ", header=0, dtype=float).values
splex_micro = pd.read_table("data/SPLEX_micro.txt", sep=" ", header=0, dtype=float).values

classes = classes.values

good_idx = []

for i, x in enumerate(classes):
    if x[0] != float('nan') and x[0] != 'nan' and not isinstance(x[0], float):
        good_idx.append(i)

d = {'LGC': 0, 'HGC':1}

classes = classes[good_idx]
classes = np.array([d[x[0]] for x in classes])

splex_host = splex_host[good_idx]
splex_micro = splex_micro[good_idx]
splex_env = splex_env[good_idx]

full_dataset = np.concatenate([splex_env, splex_micro, splex_host], axis=1)

# =========================
# Leukemia
# =========================

leukemia = pd.read_csv("data/leukemia_small.csv", header=None).values

X_leuk = np.transpose(leukemia[1:, :].astype(np.float))
Y_leuk = leukemia[0, :] == "ALL" # 1 means ALL, 0 means AML

# =========================
# Breast Cancer
# =========================

bc_data = pd.read_table("data/wdbc.data", sep=",", header=None)

X_bc = bc_data[bc_data.columns[2:]]
Y_bc = bc_data[bc_data.columns[1]]
Y_bc = Y_bc.astype("category").cat.rename_categories(range(0, Y_bc.nunique())).astype(int)

X_bc = X_bc.values
Y_bc = Y_bc.values

# Combined dataset

distrib_names = ["Breast Cancer", "leukemia", "splex_host", "splex_micro", "splex_env", "splex_full"]

datasets_name = ["Breast Cancer", "leukemia", "splex_host", "splex_micro", "splex_env", "splex_full"]

X_datasets = [X_bc, X_leuk, splex_host, splex_micro, splex_env, full_dataset]
Y_datasets = [Y_bc, Y_leuk, classes, classes, classes, classes]


# ==================================
# Datasets
# ==================================

summary = {}

for X, Y, name in zip(X_datasets, Y_datasets, datasets_name):
    summary[name] = {}

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 42)
    # Model using sklearn
    sk_model = MLPClassifier(solver="lbfgs", alpha=1E-7, hidden_layer_sizes=(10, 5))

    sk_model.fit(X_train, y_train)

    ytest_preds = sk_model.predict(X_test)

    acc = accuracy_score(y_true=y_test, y_pred=ytest_preds)

    summary[name]["test_accuracy"] = acc

    print(acc)

    keras_model = Sequential()
    keras_model.add(Dense(32, activation="relu", input_dim = X.shape[1]))
    keras_model.add(Dense(2, activation="softmax"))
    keras_model.compile(optimizer="rmsprop", loss ="sparse_categorical_crossentropy", metrics = ["accuracy"])

    history = keras_model.fit(X_train, y_train,
                              batch_size=16,
                              epochs=1,
                              verbose=1,
                              validation_split=0.1)
    score = keras_model.evaluate(X_test, y_test,
                                 batch_size=16, verbose=1)

    summary[name]["keras_testacc"] = score[1]
    summary[name]["keras_testloss"] = score[0]

    sk_cross_val = cross_val_score(sk_model, X, Y, cv=10)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    cvscores = []

    for train, test in kfold.split(X, Y):
        keras_model = Sequential()
        keras_model.add(Dense(32, activation="relu", input_dim=X.shape[1]))
        keras_model.add(Dense(2, activation="softmax"))
        keras_model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        history = keras_model.fit(X[train], Y[train],
                                  batch_size=16,
                                  epochs=30,
                                  verbose=1,
                                  validation_split=0.1)
        score = keras_model.evaluate(X[test], Y[test],
                                     batch_size=16, verbose=1)

        cvscores.append(score[1])

    summary[name]["keras_crossval"] = np.array(cvscores).mean()
    summary[name]["sklean_crossval"] = sk_cross_val.mean()

print(summary)

# Print summary:

df = pd.DataFrame.from_dict(summary, orient="index")

df.to_csv("exports/results_full.csv")



