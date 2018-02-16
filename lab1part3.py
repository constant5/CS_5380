#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 11:39:31 2018

@author: cmarks
"""

from scipy.io.arff import loadarff
with open("data/segment.arff", "r") as f:
    data, meta = loadarff(f)
print(data[0],'\n')
print('There are %d records: ' % (data.size))

X = data[meta.names()[0:-1]]
import numpy as np
X = np.asarray(X.tolist())
print(X[0])

y = data[meta.names()[-1]]
y = np.asarray(y.tolist())
print(np.unique(y))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.28)


from sklearn.tree import DecisionTreeClassifier, export_graphviz
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train)
leaves = classifier.tree_.node_count
export_graphviz(classifier, out_file = 'segment_tree.dot')

import graphviz 
graphviz.view('segment_tree.dot')

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Evaluating the Accuracy 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
from sklearn.model_selection import cross_val_score
scores= cross_val_score(classifier, X_train, y_train, cv=5)

print("Decision Tree (%d leaves) Accuracy on Segment Data: %0.2f (+/- %0.2f)\n" % (leaves, scores.mean(), scores.std() * 2))
