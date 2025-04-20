import numpy as np

def predict_anomaly(model, scaler, reading):
    reading_np = np.array(reading).reshape(1, -1)
    reading_scaled = scaler.transform(reading_np)
    prediction = model.predict(reading_scaled)
    return int(prediction[0] == -1)  # Return 1 if anomaly

def classifier(model, scaler, reading):
    reading_scaled=scaler.transform([reading])
    prediction = model.predict(reading_scaled)
    if (prediction[0]==0):
        return ("acidic environment")
    if (prediction[0]==1):
        return ("temperature drop")
    if (prediction[0]==2):
        return ("gas leak")
    if (prediction[0]==3):
        return ("overfeeding")
    if (prediction[0]==4):
        return ("underfeeding")

