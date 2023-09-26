# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('weather_next8days.csv')

X = dataset[["latitude", "longitude", "temp", "humidity", "windspeed", "sealevelpressure", "cloudcover", "visibility", "solarradiation"]]


y = dataset["precipprob"]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X.values, y.values)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[37.777,-122.42,61.8,89,17,1017.4,50,15,206.9]]))
