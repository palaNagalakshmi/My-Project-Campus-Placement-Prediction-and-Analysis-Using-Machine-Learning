import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import mean_squared_error,r2_score
import joblib
a=pd.read_csv("data.csv")
x=a.drop(columns=['RID','PlacedOrNot','Salary'])
y=a['PlacedOrNot']
x=pd.get_dummies(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
'''logisticregresssion''' 
model=LogisticRegression()
model.fit(x_train,y_train)
y_predict=model.predict(x_test)
accuracy=accuracy_score(y_predict,y_test)
print(accuracy)
aa=classification_report(y_predict,y_test)
print(aa)
'''randomforestclassifier'''
model1=RandomForestClassifier()
model1.fit(x_train,y_train)
y_predict=model1.predict(x_test)
ac_cy=accuracy_score(y_predict,y_test)
print(ac_cy)
bb=classification_report(y_predict,y_test)
print(bb)
'''linearregression'''
model2 = LinearRegression()
model2.fit(x_train,y_train)
y_predict=model2.predict(x_test)
r2=r2_score(y_test,y_predict)
print(f"R-squared:{r2}")
'''svc'''
model3=SVC(kernel="linear")
model3.fit(x_train,y_train)
y_predict=model3.predict(x_test)
ac=accuracy_score(y_predict,y_test)
print(ac)
dd=classification_report(y_predict,y_test)
print(dd)
'''navie bayes'''
model4= GaussianNB()
model4.fit(x_train, y_train)
y_predict= model4.predict(x_test)
accu=accuracy_score(y_test,y_predict)
print(accu)
ee=classification_report(y_predict,y_test)
print(ee)
joblib.dump(model2,"My_newmodel.h5")








