import pandas as pd
import numpy as np
from model_utils import train_and_save_model
from classifier_training import model_training

n_samples = 1000
timestamp = pd.date_range(start='2025-04-19', periods=n_samples, freq='10min')

# Simulate normal sensor data (without anomalies)
np.random.seed(42)
pH = np.random.normal(loc=7.2, scale=0.05, size=n_samples)             # Stable around 7.2
temperature = np.random.normal(loc=37, scale=0.5, size=n_samples)       # Stable around 37°C
gas_flow = np.random.normal(loc=5.0, scale=0.3, size=n_samples)         # Around 5 L/min
ch4_percent = np.random.normal(loc=62, scale=2, size=n_samples)         # 62% CH4
feeding_rate = np.full(shape=n_samples, fill_value=200)                 # Constant feeding

data = pd.DataFrame({
    'Timestamp': timestamp,
    'pH': pH,
    'Temperature': temperature,
    'GasFlowRate': gas_flow,
    'CH4_Percent': ch4_percent,
    'FeedingRate': feeding_rate
})

data.head()
with open('sensors_data.csv', 'w') as f:
    data.to_csv(f, index=False)

#train model
train_and_save_model('sensors_data.csv')
classifier=model_training('classifier.csv')
