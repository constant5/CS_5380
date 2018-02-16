#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 08:08:22 2018

@author: cmarks
"""
import matplotlib.pyplot as plt
from scipy.io.arff import loadarff


with open('data/mushroom.arff', 'r') as f:
    data, meta = loadarff(f)
print(data[0],'\n')
print('There are %d records: ' % (data.size))


X = data[meta.names()[0:-1]]
import numpy as np
X = np.asarray(X.tolist())


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(np.shape(X)[1]):
    X[:,i] = labelencoder_X.fit_transform(X[:, i])
X = X.astype(int)
print(X[0:5],'\n')

y = data[meta.names()[-1]]
labels = np.unique(y).astype(str)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
bin_count = np.bincount(y)
label_freq = np.array([labels, bin_count])
print('Class Frequencies: \n',label_freq)


# Splitting data into test and training set choosing a 25% sample Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.90)


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_y_pred = nb_classifier.predict(X_test)
nb_scores= cross_val_score(nb_classifier, X_train, y_train, cv=5)
print("Naive Bayes Accuracy: %0.2f (+/- %0.2f)\n" % (nb_scores.mean(), nb_scores.std() * 2))

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)
rf_scores = cross_val_score(rf_classifier, X_train, y_train, cv=5)
print("Random Forest Accuracy: %0.2f (+/- %0.2f)\n" % (rf_scores.mean(), rf_scores.std() * 2))


## Building code for ROC 

#next lines for Juyter 
# %matplotlib inline
# plt.rcParams['fontsize'] = 14


# Getiing the Binary probablity for each record

rf_classifier.fit(X_train, y_train)
rf_y_pred_prob = rf_classifier.predict_proba(X_test)[:,0]

# Plot a histogram of the probablities
plt.hist(rf_y_pred_prob, bins = 8 )
plt.title('Histogram of predicted probablities')
plt.xlabel('Predicted Probablity of Edible Mushrooms')
plt.ylabel('Frequency')

# Lowering the threshold to make the model more sensitive
from sklearn.preprocessing import binarize
rf_y_pred_class = binarize(rf_y_pred_prob, 0.3)[0]

from sklearn.metrics import confusion_matrix

cm1 = confusion_matrix(y_test,rf_y_pred)
cm2 = confusion_matrix(y_test,rf_y_pred_class)

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, rf_y_pred_prob, pos_label = 0)
plt.plot(fpr,tpr,'ro')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('ROC Curve for RF Mushroom Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rte')
plt.grid(True)
