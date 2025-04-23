import pandas as pd
import numpy as np
from model_utils import train_and_save_model
from classifier_training import model_training
from prediction import detect_anomalies
from generate_data import generate_without_anomalies, add_anomalies 


data=generate_without_anomalies()
# Save the generated dataset in a file 
with open('./data/sensors_data.csv', 'w') as f:
    data.to_csv(f, index=False)

# Train the model
train_and_save_model('./data/sensors_data.csv')


# Generate dataset with anomalies 
data=add_anomalies(data)
print('correct')
print(data[data['is_anomaly']==1])

# Save the data with anomalies 
with open('./data/sensors_data_with_anomalies.csv', 'w') as f:
    data.to_csv(f, index=False)

# Predictions 
data.drop(columns='is_anomaly',inplace=True)
predictions = detect_anomalies(data)

data['Anomaly'] = predictions
data['Anomaly'] = data['Anomaly'].map({1: 0, -1: 1})  # 1 for anomaly 
#data.to_csv("preds.csv", index=False)
print("preds")
print(data[data['Anomaly']==1])


classifier=model_training('classifier.csv')
