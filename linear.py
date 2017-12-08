import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv('MT_Train.csv')
pred_data = pd.read_csv('MT_Test.csv')

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
x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]


# prepare test data
x_test = pred_data.iloc[:, 1:]

mx = LinearRegression()
mx.fit(x_train.values, y_train.values)

y_pred = {}
y_predY=[]
y_predS=[]
y_predNum = mx.predict(x_test)
print(y_predNum)
y_predNum = y_predNum.round().astype(int)

for i in y_predNum:
    if i==0:
        y_predY.append('no')
    else:
        y_predY.append('yes')
leng = len(y_predNum)
print(leng)
for i in range(0,leng):
    y_predS.append(i)
y_pred['sampleId'] = y_predS
y_pred['y'] = y_predY
data = pd.DataFrame(y_pred)
data.to_csv('MT_y_linearpred.csv', index=False)
