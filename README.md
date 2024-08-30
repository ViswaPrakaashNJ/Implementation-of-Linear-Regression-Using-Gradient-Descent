# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step-1:start
step-2:Import the required library and read the dataframe.
step-3:Write a function computeCost to generate the cost function.
step-4:Perform iterations og gradient steps with learning rate.
step-5:Plot the Cost function using Gradient Descent and generate the required graph.
step-6:stop. 

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: VISWA PRAKAASH N J
RegisterNumber:  212223040246
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")

```

## Output:
X values

![326158720-9bf60fa0-af77-42cf-a31d-ea158d181abb](https://github.com/user-attachments/assets/91483ce0-9e67-4de5-88f0-5f958a2d171c)

y values

![326158727-3be61adc-d7e3-4d8c-97b4-3e0ac9547133](https://github.com/user-attachments/assets/b24bac2f-e438-481f-bc77-84bc72c6e9e2)

X Scaled values

![326158733-c0b84bd5-3d9e-45f2-8f24-7908acecff74](https://github.com/user-attachments/assets/bd67886d-464d-414a-8f55-698f649f68c6)

y Scaled values

![326158906-9d167518-03cc-4e2c-b206-93ac4cf31ba6](https://github.com/user-attachments/assets/9977cff5-1d78-47ad-83de-4106a15a1c79)

Predicted value

![326158770-9c78ef45-c44a-4873-8ff1-262becf88045](https://github.com/user-attachments/assets/96a97589-894a-4711-857c-7ab0589d9836)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
