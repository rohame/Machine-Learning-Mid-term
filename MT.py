# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:13:33 2017

@author: Jim
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# load raw data
train_data = pd.read_csv('MT_Train.csv')
pred_data = pd.read_csv('MT_Test.csv')

# do preprocessing
strIndex = train_data.columns[train_data.dtypes == object]
for i in strIndex:
    le = LabelEncoder()
    le.fit(train_data[i])
    train_data[i] = le.transform(train_data[i])
    try:
        pred_data[i] = le.transform(pred_data[i])
    except:
        pass


# prepare train data
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]


# prepare test data
X_pred = pred_data.iloc[:, 1:]

## 调这里
clf = RandomForestClassifier(n_estimators=12000, n_jobs=-1, class_weight={0:5})
clf.fit(X, y)


y_pred = clf.predict_proba(X_pred)
probability = []
length = len(y_pred)
for i in range(0,length):
    y = y_pred[i]
    prob = y[-1]+0.025
    prob = prob.round().astype(int)
    probability.append(prob)


le = LabelEncoder()
le.fit(["yes", "no"])
probability = le.inverse_transform(probability)

data = pd.DataFrame({'sampleId':range(0,len(probability)),'y':probability})
data.to_csv('RandomForest.csv', index=False)
