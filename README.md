# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. To train, initialize theta and iteratively update it using gradient descent.
2. For preprocessing, read and scale data.
3. For modeling, train the linear regression model.
4. To predict data values, scale a new data and predict.
5. Print the prediction.

## Program & Output :
```
/*
Program to implement the linear regression using gradient descent.
Developed by: ALIYA SHEEMA
RegisterNumber: 212223230011
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df=pd.read_csv("student_scores.csv")
print(df.head())
print(df.tail())
```
![image](https://github.com/user-attachments/assets/33cb1929-40d1-4a1a-92f3-28348d3f59be)

```
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
```

![image](https://github.com/user-attachments/assets/65fb4793-6b81-4413-beac-ee11b85e9ea6)

```
x.shape
```

![image](https://github.com/user-attachments/assets/9eec4581-ecfb-4328-bd23-987d78398e97)

```
y.shape
```

![image](https://github.com/user-attachments/assets/e8729171-45ab-479e-aaa9-f9c970a61b52)

```
m=0
c=0
L=0.001 # learning rate
epochs=5000 # No.of iterations to be performed
n=float(len(x))
error=[]
# Performing Gradient Descent
for i in range(epochs):
  y_pred = m*x + c
  D_m = (-2/n)*sum(x*(y-y_pred))
  D_c = (-2/n)*sum(y-y_pred)
  m = m-L*D_m
  c = c-L*D_c
  error.append(sum(y-y_pred)**2)
print(m,c)
type(error)
print(len(error))
plt.plot(range(0,epochs),error)
```

![image](https://github.com/user-attachments/assets/00684343-9a68-45c8-b260-122f1e4c7ab5)

![image](https://github.com/user-attachments/assets/31fef9b3-9cd1-4a47-8422-4c3a837d02e7)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
