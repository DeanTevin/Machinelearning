from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from math import sqrt


penguins = pd.read_csv('trainingdata.csv')

# Ordinal feature encoding
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = penguins.copy()
target = 'status'

target_mapper = {'suspect':0, 'good':1}
def target_encode(val):
    return target_mapper[val]

df['status'] = df['status'].apply(target_encode)

# Separating X and y
X = df.drop(['status','cycle','unit'], axis=1)
y = df['status']

# # Build random forest model
# from sklearn.ensemble import RandomForestClassifier

# #build model
# X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size=0.7)
# sc_X = StandardScaler()
# X_trainscaled=sc_X.fit_transform(X_train)
# X_testscaled=sc_X.transform(X_test)

#build model
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y ,random_state=1)
sc_X = StandardScaler()
X_trainscaled=sc_X.fit_transform(X_train)
X_testscaled=sc_X.transform(X_test)

clf = MLPClassifier(hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1,max_iter=1000).fit(X_trainscaled, y_train)

clf.fit(X, y)
# clf = MLPClassifier(solver="lbfgs",hidden_layer_sizes=(256,128,64,32),activation="relu",random_state=1,max_iter=1000).fit(X_trainscaled, y_train)

# from sklearn import metrics
# predict_test = clf.predict(X_test)
# print(metrics.accuracy_score(y_test, predict_test))
# print(sqrt(metrics.mean_squared_error(y_test, predict_test)))
# print(metrics.mean_absolute_error(y_test, predict_test))

# Saving the model
import pickle
pickle.dump(clf, open('data.pkl', 'wb'))
