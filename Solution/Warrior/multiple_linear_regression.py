

# Multiple Linear Regression


"""## Importing the libraries"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset = pd.read_csv('Train_dataset_.csv')
dataset.drop(['Stock Index','Index','Industry',],axis=1,inplace=True);
dataset.head()
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


print(dataset.dtypes);


#handling missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[: , :-1])
X[: , :-1] = imputer.transform(X[: , :-1])

print(y)

"""## Encoding categorical data"""



print(X)

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X[: , :-1], y, test_size = 0.2, random_state = 0)

"""## Training the Multiple Linear Regression model on the Training set"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

"""## Predicting the Test set results"""

y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

"""# **Score Card for the model.**"""

from sklearn.metrics import r2_score
print(r2_score (y_test,y_pred)*100);

"""# **predicting for 10th Aug**"""

datatest = pd.read_csv('Train_dataset_.csv')
datatest.drop(['Stock Index','Index','Industry',],axis=1,inplace=True);
dataset.head()
X = dataset.iloc[:, :-1].values
regressor.predict([[12765.84,32.38,45.35,1208599,5.05,-0.43,0.052,1,0.61,9.2]]) #input all the variables here and get result seprate for each stock.
