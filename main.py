# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 22:29:46 2020

@author: kingslayer
"""

#POLYNOMIAL REGRESSION
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r"Position_Salaries.csv")

# creating matrix of features
X=dataset.iloc[:,1:2].values

#creating dependant variable vector
y=dataset.iloc[:,-1].values

#splitting into training and test set
"""from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)"""

#creating linear regression model
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)

#creating polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)
"""now again fitting linear model to X_poly"""
lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

#visualing the results of linear model
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X))
plt.title("Position vs Salary(Linear Model)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualising the results of polynomial model
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color="red")
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color="blue")
plt.title("Position vs Salary(Polynomial Model)")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#predicting new results with linear model
lin_reg.predict([[6.5]])

#predicting new results with linear model
lin_reg2.predict(poly_reg.fit_transform([[8]]))