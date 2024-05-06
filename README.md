# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6. Obtain the graph.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PYNAM VINODH
RegisterNumber:  212223240131
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Placement_Data.csv')
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
Y
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(Y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5,1,0)
    return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(y_pred)
print(Y)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
*/
```

## Output:
### Dataset:
![image](https://github.com/PYNAMVINODH/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742678/2551d4ae-f5df-4c36-8f29-668c16f4c72e)

### Categories:
![image](https://github.com/PYNAMVINODH/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742678/ce594ffd-1beb-4b94-b6b4-ff68fa6206fc)

![image](https://github.com/PYNAMVINODH/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742678/30c1b972-bc35-4e4f-9429-3607769cb6d7)

### X&Y values:
![image](https://github.com/PYNAMVINODH/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742678/5953bdf8-0fb4-46a6-8c00-a733844d8281)

### Accurcay and y_pred:
![image](https://github.com/PYNAMVINODH/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/145742678/1256b91b-d8e1-4f0c-a7a5-0a14c9c66342)







## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
