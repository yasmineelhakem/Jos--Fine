import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

def train_and_save_model(filepath='sensor_data.csv'):
    data = pd.read_csv(filepath)
    X = data[['pH', 'Temperature', 'GasFlowRate', 'CH4_Percent', 'FeedingRate']]

    # Data Normalization 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=0.01,
        random_state=42
    )
    model.fit(X_scaled)

    # Save the trained Isolation Forest model and the scaler for later use
    joblib.dump(model, './models/isolation_forest_model.pkl')
    joblib.dump(scaler, './models/scaler.pkl')
    print(" Model and scaler saved.")

def load():
    model = joblib.load('./models/isolation_forest_model.pkl')
    scaler = joblib.load('./models/scaler.pkl')
    classifier= joblib.load('./models/classifier.pkl')
    return model, scaler, classifier
