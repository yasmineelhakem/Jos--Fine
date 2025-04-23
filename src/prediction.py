import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def detect_anomalies(data, contamination=0.01):
    features = ['pH', 'Temperature', 'GasFlowRate', 'CH4_Percent', 'FeedingRate']
    X = data[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=contamination,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    iso_forest.fit(X_scaled)
    predictions = iso_forest.predict(X_scaled)  # -1 = anomaly, 1 = normal
    prediction_df = pd.DataFrame(predictions, columns=['Prediction'])
    print("Anomaly counts:")
    print(prediction_df.value_counts())

    return predictions