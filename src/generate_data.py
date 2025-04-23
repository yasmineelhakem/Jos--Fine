import random
import numpy as np
import pandas as pd

def generate_without_anomalies():
    n_samples = 1000
    timestamp = pd.date_range(start='2025-04-19', periods=n_samples, freq='10min')

    # Simulate normal sensor data (without anomalies) by generating a dataset due to the absence of datasets 
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

    return data

def add_anomalies(data):

    n_anomalies = 10
    anomaly_indices = np.random.choice(1000, n_anomalies, replace=False)
    data.loc[anomaly_indices[:3], 'pH'] -= np.random.uniform(0.8, 1.2, size=3) # Acidification event: pH drops
    data.loc[anomaly_indices[3:6], 'Temperature'] -= np.random.uniform(5, 8, size=3) # Temperature shock: temp drops
    data.loc[anomaly_indices[6:], 'GasFlowRate'] -= np.random.uniform(2, 3, size=4) # Gas flow crash: flow rate drops
    data['is_anomaly'] = 0
    data.loc[anomaly_indices, 'is_anomaly'] = 1
    
    data[data['is_anomaly']==1].head(5)

    return data
def random_sensor_reading():
    if random.random() < 0.05:  # 5% anomaly chance
        anomaly_type = random.choice(['acid', 'cold', 'gas_loss', 'ch4_loss', 'overfeed'])

        if anomaly_type == 'acid':
            reading = [random.uniform(5.5, 6.4), random.uniform(35, 38), random.uniform(3.5, 5.5), random.uniform(55, 65), random.uniform(180, 230)]
        elif anomaly_type == 'cold':
            reading = [random.uniform(6.8, 7.5), random.uniform(28, 32), random.uniform(3.5, 5.5), random.uniform(55, 65), random.uniform(180, 230)]
        elif anomaly_type == 'gas_loss':
            reading = [random.uniform(6.8, 7.5), random.uniform(35, 38), random.uniform(1.0, 2.0), random.uniform(55, 65), random.uniform(180, 230)]
        elif anomaly_type == 'ch4_loss':
            reading = [random.uniform(6.8, 7.5), random.uniform(35, 38), random.uniform(3.5, 5.5), random.uniform(30, 45), random.uniform(180, 230)]
        elif anomaly_type == 'overfeed':
            reading = [random.uniform(6.8, 7.5), random.uniform(35, 38), random.uniform(3.5, 5.5), random.uniform(55, 65), random.uniform(250, 300)]

        return reading, anomaly_type

    else:
        reading = [random.uniform(6.8, 7.5), random.uniform(35, 38), random.uniform(3.5, 5.5), random.uniform(55, 65), random.uniform(180, 230)]
        return reading, None
