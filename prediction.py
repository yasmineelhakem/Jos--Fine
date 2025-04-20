#prédire les anomalies avec Isolation Forest
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(filepath='sensor_data_with_anomalies.csv'):
    data = pd.read_csv(filepath, parse_dates=['Timestamp'])

    X = data[['pH', 'Temperature', 'GasFlowRate', 'CH4_Percent', 'FeedingRate']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.01,
        random_state=42
    )
    predictions = model.fit_predict(X_scaled)

    # Add results to DataFrame
    data['Anomaly'] = np.where(predictions == -1, 1, 0)

    return data