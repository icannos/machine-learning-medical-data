import pandas as pd
import graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# For reproducibility
np.random.seed(123456)


# Load the data

data_diabete = pd.read_table("data/patients_data.txt", sep="\t", header=None)
classes_diabete = pd.read_table("data/patients_classes.txt", sep="\t", header=None)

# ============================
# Decision Tree
# ============================

# Create and fit classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(data_diabete, classes_diabete)

# Visualise tree

features_names = ['age', 'hba1c', 'insuline taken', 'other drugs taken']
classes = ['DR', 'NDR']

dot_data = tree.export_graphviz(clf,
                                 #out_file='exports/diabete_tree1.pdf',
                                 feature_names=features_names,
                                 class_names=classes,
                                 filled=True, rounded=True,
                                 special_characters=True
                                 )
graph = graphviz.Source(dot_data)
graph.render("exports/classification_tree_diabete_remission")

# ============================
# Random Forest
# ============================

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = clf.fit(data_diabete, classes_diabete.values[:, 0])

df = pd.DataFrame([clf.feature_importances_], columns=features_names)
df.name = "feature importances"
df.to_csv("exports/random_forest_feature_importance.csv", sep=';')



