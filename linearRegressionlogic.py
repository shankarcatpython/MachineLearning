import pandas as pd
import numpy as np

# Load data
data = pd.read_csv('data.csv')

X = data['predictor_variable'].values.reshape(-1,1)
Y = data['target_variable'].values.reshape(-1,1)

coefficient = np.zeros((2,1))

coefficient[0] = 1
coefficient[1] = 2
increment = 0.001

print(coefficient)

for value in range(0,(len(X)//10)):
    deviation = abs(Y[value] - (X[value] * coefficient[0] + coefficient[1]))
    adjusted = deviation * increment 
    coefficient[0] +=  adjusted
    coefficient[1] +=  adjusted
    print(Y[value],(X[value] * coefficient[0] + coefficient[1]),adjusted)

print(coefficient)

'''
for value in range(0,len(X)):
    print(Y[value] , ((X[value] * coefficient[0] )+ coefficient[1]))
    break
'''