import joblib
import pandas as pd
model=joblib.load("My_newmodel.h5")
f=model.feature_names_in_
a=int(input("enter age"))
b=input("enter gender")
c=input("enter stream")
d=int(input("enter HistoryOfBacklogs"))
e=int(input("enter Internships"))
f_b=int(input("enter Btech_CGPA"))
g=int(input("enter SSLC_Percentage"))
h=int(input("enter PUC_Percentage"))
i=int(input("enter Hostel"))
d1={"Age":a,"gender":"b","stream":"c","HistoryOfBacklogs":d,"Internships":e,"Btech_CGPA":f_b,"SSLC_Percentage":g,"PUC_Percentage":h,"Hostel":i}
d2=pd.DataFrame([d1])
d2=pd.get_dummies(d2)
d2=d2.reindex(columns=f,fill_value=0)
p=model.predict(d2)
print(p)
if p>=1:
    print("Placed")
else:
    print("not placed")
