import numpy as np

def predict_anomaly(model, scaler, reading):
    reading_np = np.array(reading).reshape(1, -1)
    reading_scaled = scaler.transform(reading_np)
    prediction = model.predict(reading_scaled)
    return int(prediction[0] == -1)  # Return 1 if anomaly
