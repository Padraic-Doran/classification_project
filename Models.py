#!/usr/bin/env python
# coding: utf-8

# In[1]:


import env
import prep
# import models
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from os import path
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import math
import split_scale
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from graphviz import Source
from scipy import stats


# In[2]:


def decisiontree(numeric_X_train, y_train):
    clf = DecisionTreeClassifier(max_depth=2, max_features = 3, random_state=123)
    clf.fit(numeric_X_train, y_train)
    y_pred = clf.predict(numeric_X_train)
    y_pred
    y_pred_proba = clf.predict_proba(numeric_X_train)
    y_pred_proba
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(numeric_X_train, y_train)))
    confusion_matrix(y_train, y_pred)
    print(classification_report(y_train, y_pred))
    dot_data = export_graphviz(clf, out_file=None) 
    graph = Source(dot_data) 

    graph.render('telco_decision_tree', view=True)
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
     .format(clf.score(numeric_X_test, y_test)))
    return clf.score(numeric_X_train, y_train), clf.score(numeric_X_test, y_test)


# In[3]:


def LogRegression(numeric_X_train, y_train):
    logit = LogisticRegression(C=1, class_weight={1:2}, random_state = 123, solver='saga')
    logit.fit(numeric_X_train, y_train)
    print('Coefficient: \n', logit.coef_)
    print('Intercept: \n', logit.intercept_)
    y_pred = logit.predict(numeric_X_train)
    y_pred_proba = logit.predict_proba(numeric_X_train)
    print('Accuracy of Logistic Regression classifier on training set: {:.2f}'
         .format(logit.score(numeric_X_train, y_train)))
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))
    print('Accuracy of Logistic Regression classifier on test set: {:.2f}'
         .format(logit.score(numeric_X_test, y_test)))
    return logit.score(numeric_X_train, y_train), logit.score(numeric_X_test, y_test)


# In[4]:


def RandForest(numeric_X_train, y_train):
    rf = RandomForestClassifier(bootstrap=True, 
                            class_weight=None, 
                            criterion='gini',
                            min_samples_leaf=3,
                            n_estimators=100,
                            max_depth=2, 
                            random_state=123)
    rf.fit(numeric_X_train, y_train)
    print(rf.feature_importances_)
    y_pred = rf.predict(numeric_X_train)
    y_pred_proba = rf.predict_proba(numeric_X_train)
    print('Accuracy of random forest classifier on training set: {:.2f}'
         .format(rf.score(numeric_X_train, y_train)))
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))
    print('Accuracy of random forest classifier on test set: {:.2f}'.format(rf.score(numeric_X_test, y_test)))
    return rf.score(numeric_X_train, y_train), rf.score(numeric_X_test, y_test)


# In[5]:


def KNN(numeric_X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    knn.fit(numeric_X_train, y_train)
    y_pred = knn.predict(numeric_X_train)
    y_pred_proba = knn.predict_proba(numeric_X_train)
    print('Accuracy of KNN classifier on training set: {:.2f}'
         .format(knn.score(numeric_X_train, y_train)))
    print(confusion_matrix(y_train, y_pred))
    print(classification_report(y_train, y_pred))
    print('Accuracy of KNN classifier on test set: {:.2f}'
         .format(knn.score(numeric_X_test, y_test)))
    return knn.score(numeric_X_train, y_train), knn.score(numeric_X_test, y_test)


# In[ ]:




