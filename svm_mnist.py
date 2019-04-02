import pandas as pd
import numpy as np
from sklearn import svm

train = pd.read_csv('mnist_train.csv')
test = pd.read_csv('mnist_test.csv')

x_train = train.iloc[0:60000,1:].values
y_train = train.iloc[0:60000,0].values
x_test = test.iloc[0:10000,1:].values
y_test = test.iloc[0:10000,0].values

x_train[x_train>0]=1
x_test[x_test>0]=1

cl = svm.SVC(C=200,kernel='rbf',gamma=0.01,cache_size=8000,probability=False)
cl.fit(x_train,y_train)

y_predict = cl.predict(x_test)

count=0

for a in range(0,10000):
    if(y_predict[a]==y_test[a]):
        count=count+1
percent_acc = count/10000*100

print("the accuracy is: ")
print(percent_acc)
#accuracy is 98.1% 

