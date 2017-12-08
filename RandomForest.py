# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 21:55:11 2017

@author: Jim
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# load raw data
train_data = pd.read_csv('MT_Train.csv')

# do preprocessing
strIndex = train_data.columns[train_data.dtypes == object]
for i in strIndex:
    le = LabelEncoder()
    le.fit(train_data[i])
    train_data[i] = le.transform(train_data[i])

# prepare train data
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, class_weight={0:10})
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred))



