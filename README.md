# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vigneshkumar V
RegisterNumber: 212220220054
*/
```
import pandas as pd

data = pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["salary"] = le.fit_transform(data["salary"])

data.head()

x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()

y = data["left"]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(criterion = "entropy")

dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)

from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)

accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])

## Output:
1.Data Head

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113573894/161eca20-3441-41d5-837c-30dc92f4fb31)

2.Data Info

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113573894/90a84e0c-36de-4461-aaea-5de2eb48b8ab)

3.Data isnull

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113573894/4a864da4-dba1-40d3-8d4e-87a6cbafab3c)

4.Data Left

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113573894/5b3004c5-2cdd-4293-a8a5-1ba87e9932e8)

5.X Head

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113573894/bc84f15b-8107-4c1f-9772-49f63b3aa771)

6.Data fit

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113573894/4e88e604-26f5-49e5-a90a-cbe4a98b8e4a)

7.Accuracy

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113573894/d0e1594b-cae4-4542-96ff-bd4b43d11288)

8.Predicted Values

![image](https://github.com/VigneshKumar1009/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113573894/8811f2d1-bb43-4021-a920-b60b248bb091)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
