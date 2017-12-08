import numpy as np
import scipy as sp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

f1 = 'MT_Train.csv'
f2 = 'MT_Test.csv'
df1 = pd.read_csv(f1, index_col=False)
df2 = pd.read_csv(f2, index_col=False)

strIndex1 = df1.columns[df1.dtypes == object]
for Index in strIndex1:
    le = LabelEncoder()
    le.fit(df1[Index])
    df1[Index] = le.transform(df1[Index])

strIndex2 = df2.columns[df2.dtypes == object]
for Index in strIndex2:
    le = LabelEncoder()
    le.fit(df2[Index])
    df2[Index] = le.transform(df2[Index])

xlst, ysgn = 'age job marital education default housing loan contact month day_of_week duration campaign pdays previous ' \
             'poutcome emp.var.rate cons.price.idx cons.conf.idx euribor3m nr.employed'.split(), 'y'

x_train, y_train = df1[xlst], df1[ysgn]
x_test = df2[xlst]

mx = LogisticRegression()
mx.fit(x_train.values, y_train.values)

y_pred = {}
y_predY=[]
y_predS=[]
y_predNum = mx.predict_proba(x_test)
probability = []
length = len(y_predNum)
for i in range(0,length):
    y = y_predNum[i]
    prob = y[-1]+0.15
    prob = prob.round().astype(int)
    probability.append(prob)

#probability = probability.round().astype(int)

for i in probability:
    if i==0:
        y_predY.append('no')
    else:
        y_predY.append('yes')
leng = len(probability)
print(leng)
for i in range(0,leng):
    y_predS.append(i)
y_pred['sampleId'] = y_predS
y_pred['y'] = y_predY
data = pd.DataFrame(y_pred)
data.to_csv('MT_y_logisticpred.csv', index=False)
