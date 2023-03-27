# Import libraries
import pandas as pd
import numpy as np
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math

# Load data
data = pd.read_csv('data.csv')

# Define the predictor and target variables
X = data['predictor_variable'].values.reshape(-1,1) # predictor variable as a numpy array
y = data['target_variable'].values.reshape(-1,1) # target variable as a numpy array

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fit the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = regressor.predict(X_test)

# Evaluate the model using metrics such as mean squared error and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean squared error: ", mse)
print("R-squared: ", r2)
print(regressor.coef_)

'''
for i in range(0,len(y_test)):
    print(math.sqrt(y_test[i]),y_test[i],y_pred[i])
'''