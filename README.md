# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
```
Developed by: SUDHARSAN S
Reg No:  212224040334
```
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.

2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.

## Program:
```
import pandas as pd
data=pd.read_csv('Employee.csv')

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data[["left"]]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:

head:

<img width="1236" height="225" alt="image" src="https://github.com/user-attachments/assets/44e91a79-909e-4fb9-b98d-34b7fc21dda6" />

info:

<img width="809" height="324" alt="image" src="https://github.com/user-attachments/assets/c7b19a99-7649-4b9b-bbdd-01592e0ac650" />

isnull:

<img width="781" height="223" alt="image" src="https://github.com/user-attachments/assets/c35f3b7f-d896-4612-a39b-fac456a7fad2" />

left:

<img width="571" height="104" alt="image" src="https://github.com/user-attachments/assets/93933423-5c32-44e6-b96a-410a887e1e2e" />
 
head:

<img width="1218" height="219" alt="image" src="https://github.com/user-attachments/assets/097d2366-ef4e-4221-bfd1-30afc9c70960" />

x.head:

<img width="1237" height="228" alt="image" src="https://github.com/user-attachments/assets/0270340a-f98f-42b3-a3c5-a29d258d040a" />

y.head:

<img width="542" height="235" alt="image" src="https://github.com/user-attachments/assets/ebd701b7-cbb1-4358-a31b-511addf3847b" />

y_pred:

<img width="927" height="101" alt="image" src="https://github.com/user-attachments/assets/0cb22498-121c-4a7e-aa0f-bf9efc048ed8" />

accuracy:

<img width="699" height="57" alt="image" src="https://github.com/user-attachments/assets/a2b796a0-6125-4b21-bed3-447ad930f293" />

confusion:

<img width="773" height="57" alt="image" src="https://github.com/user-attachments/assets/39407ed5-98e3-4b26-9ba8-37c86ae105f6" />

classification report:

<img width="775" height="178" alt="image" src="https://github.com/user-attachments/assets/c09f6f6e-57e1-488e-8838-11a42e2e8af2" />

predict:
<img width="1228" height="113" alt="image" src="https://github.com/user-attachments/assets/850a2aa2-896c-4db2-8dfb-53f0d0cec56b" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
