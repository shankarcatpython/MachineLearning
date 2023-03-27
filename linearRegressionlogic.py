import pandas as pd
import random

random.seed(0)

# Training a model 

def modeltraining(X,Y,coefficient=[],increment=0.1,iteration=10):

    coefficient.append(increment)

    for value in range(0,iteration):
        random_value = random.randint(0, len(X))
        input_value=X[random_value]
        output_value=Y[random_value]
        predicted_value= round(( input_value * coefficient[0]),2)
        deviation = round((predicted_value - output_value),2)
        adjusted = round((input_value * increment),2)

        if output_value > predicted_value:
            coefficient[0] += round(adjusted)
        else:
            coefficient[0] -= round(adjusted)

        coefficient[0] = round(coefficient[0])
  

        #print(output_value,predicted_value,round(predicted_value-output_value),round((predicted_value/output_value)*100))


    return(coefficient)

def predict(source,coefficient):
    predicted_value= round((source * coefficient[0]),2)
    return(predicted_value)

if __name__ == '__main__':

# Raw data to be used for modeling and prediction
    data = pd.read_csv('data.csv')

    X = data['predictor_variable'].tolist()
    Y = data['target_variable'].tolist()

    coefficients = modeltraining(X,Y,[],0.1,1000)   
    
    for i in range(1,1000):
        print(X[i],Y[i],predict(X[i],coefficients))