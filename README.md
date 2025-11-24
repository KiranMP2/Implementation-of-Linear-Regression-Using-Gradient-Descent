# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required library and read the dataframe

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the cost function using Gradient Descent and generate the required graph.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: KIRAN MP
RegisterNumber: 212224230123
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    X=np.c_[np.ones(len(X1)), X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())
```
```
print('Name:KIRAN MP')
print("Register No: 212224230123")
X=(data.iloc[1:, :-2].values)
print()
print(X)
print()
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:, -1].values).reshape(-1,1)
print(y)
print()
```
```
print('Name:KIRAN MP')
print("Register No: 212224230123")
X=(data.iloc[1:, :-2].values)
print()
print(X)
print()
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:, -1].values).reshape(-1,1)
print(y)
print()
```
```
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print()
print('Name:KIRAN MP')
print("Register No: 212224230123")
print()
print(X1_Scaled)
print()
print(Y1_Scaled)
print()
```
```
print('Name:KIRAN MP')
print("Register No: 212224230123")
theta=linear_regression(X1_Scaled, Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled =scaler.fit_transform(new_data)
prediction =np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```
## Output:
<img width="653" height="112" alt="Screenshot 2025-11-24 153605" src="https://github.com/user-attachments/assets/72ed4b2d-9be8-46a6-b79b-af6e6ef011ad" />

<img width="498" height="492" alt="Screenshot 2025-11-24 153614" src="https://github.com/user-attachments/assets/a6fe22c6-33ce-471a-b438-44998709a9c0" />

<img width="584" height="265" alt="Screenshot 2025-11-24 153624" src="https://github.com/user-attachments/assets/8d4c1bac-1886-4b92-a5dd-2fdb32942990" />

<img width="348" height="65" alt="Screenshot 2025-11-24 153633" src="https://github.com/user-attachments/assets/a285b1de-6b28-4e78-8310-21255eead044" />
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
