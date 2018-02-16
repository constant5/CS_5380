# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:19:15 2018

@author: crm0376
"""

# Importing Data and Inspect

from scipy.io.arff import loadarff
with open('data/splice.arff', 'r') as f:
    data, meta = loadarff(f)
#print(meta)
print('there are %d data points: ' % (data.size))
#print(data)

# Preproccesing 
# Slice out feautre set
X = data[meta.names()[1:-1]]
#print(X)

# Convert X data to numpy array and then cast to integers
import numpy as np
X = np.asarray(X.tolist())

# Get set of unique features
features = np.unique(X)
print('Features: ',features)

# Encoding the data set three ways

# using a built in print function
# X = X.view(dtype='uint8')

# 'Manual' Encoding with for loop
#n=1
#for i in np.nditer(feaures):
#    X[X==i] = n
#    n = n+1
#X = X.astype(int)
#print(X.dtype)   

# Using the sklearn LabelEncoder Class
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
labelencoder_X.fit(features)
for i in range(np.shape(X)[1]):
    X[:,i] = labelencoder_X.transform(X[:, i])
X = X.astype(int)

# Extract dependant vairiable
y = data[meta.names()[-1]]
#print(Y)
labels = np.unique(y).astype(str)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
bin_count = np.bincount(y)
label_freq = np.array([labels, bin_count])
print('Label Frequencies: ',label_freq)


# Create Niave Bayes Classifier and Cross Validate w/ folds = 5,10,15
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
from sklearn.model_selection import cross_val_score

scores_5 = cross_val_score(classifier, X, y, cv=5)
print(scores_5.mean())
scores_10 = cross_val_score(classifier, X, y, cv=10)
print(scores_10.mean())
scores_15 = cross_val_score(classifier, X, y, cv=15)
print(scores_15.mean())

# Splitting data into test and training set choosing a 10% sample Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fitting the classifier
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and Calculatng Accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
nb_accuracy = (cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
nb_error = 1 - nb_accuracy 

# Format the output
print('The Naive Bayes Classifier with 0.1 test split has a single test accuracy rate of %.2f\n \
      \t with an error rate of %.2f and a k-fold cross validation score of %.2f' % (nb_accuracy, nb_error, scores_10.mean()))

# Create Rnndom Forest Classifier and Cross Validate w/ folds = 5,10,15
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')

from sklearn.model_selection import cross_val_score
scores_5 = cross_val_score(classifier, X, y, cv=5)
#print(scores_5.mean())
scores_10 = cross_val_score(classifier, X, y, cv=10)
#print(scores_10.mean())
scores_15 = cross_val_score(classifier, X, y, cv=15)
#print(scores_15.mean())

# Splitting data into test and training set choosing a 10% sample Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

# Fitting the classifier
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix and Calculatng Accuracy
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
rf_accuracy = (cm[0,0]+cm[1,1]+cm[2,2])/np.sum(cm)
rf_error = 1 - rf_accuracy 

# Format the output
print('The Random Forest Classifier with 0.1 test split has a single test accuracy rate of %.2f\n \
      \t with an error rate of %.2f and a k-fold cross validation score of %.2f' % (rf_accuracy, rf_error, scores_10.mean()))
















