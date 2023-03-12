import pandas as pd
import numpy as np
import random

# Load data
data = pd.read_csv('data.csv')

X = data['predictor_variable'].tolist()
Y = data['target_variable'].tolist()

coefficient = []
coefficient.append(0.1)
coefficient.append(0.1)

increment = 0.01

for value in range(0,10):
    random_value = random.randint(0, len(X))
    temp=X[random_value]
    predicted_value= round((temp * coefficient[0] + coefficient[1]),2)
    deviation = round((predicted_value - temp),2)
    adjusted = round((deviation * increment),2)

    if  adjusted >= 0:
        coefficient [0] -= 0.01
        coefficient [1] -= 0.01
    else:
        coefficient [0] += 0.01
        coefficient [1] += 0.01

    print(random_value,temp,predicted_value,deviation,adjusted)