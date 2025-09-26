# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
#Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#Developed by: Rihan Ahamed
#RegisterNumber:  212224040276

import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
### 1.Placement Data
<img width="1276" height="221" alt="image" src="https://github.com/user-attachments/assets/6ad6cee4-d675-4366-a79c-f39954dcd44e" />
### 2.Salary Data
<img width="1068" height="209" alt="image" src="https://github.com/user-attachments/assets/d0958b4f-074b-41f1-b393-79fcdfc65b70" />
### 3.Checking the null function()
<img width="249" height="369" alt="image" src="https://github.com/user-attachments/assets/40a0a22f-ccfd-45d1-a495-3409b2f137f2" />
### 4.y_prediction array
<img width="935" height="73" alt="image" src="https://github.com/user-attachments/assets/5224c1bd-d725-43d0-94d0-5a67cfede760" />
### 5.Accuracy value
<img width="396" height="42" alt="image" src="https://github.com/user-attachments/assets/676d6a17-3bdd-4976-a913-dddf3dff7fcb" />
### 6.Confusion matrix
<img width="188" height="79" alt="image" src="https://github.com/user-attachments/assets/c70ac320-34e8-4a61-9974-4ee9a26c689b" />
### 7.Classification Report
<img width="628" height="223" alt="image" src="https://github.com/user-attachments/assets/c5439cc7-6261-4763-8e57-5595a07e9fdb" />







## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
