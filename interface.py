import pandas as pd
from datetime import datetime
import streamlit as st
import plotly.express as px
from prediction import detect_anomalies
from random_sensors_data import random_sensor_reading
from predictor import predict_anomaly, classifier
import joblib
import time

# Load data and models
data = detect_anomalies()
model = joblib.load('isolation_forest_model.pkl')
classifier_model = joblib.load('classifier.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Biogas Sensor Anomaly Detection Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(data))
col2.metric("Anomalies", data['Anomaly'].sum())
col3.metric("Anomaly Rate (%)", round(100 * data['Anomaly'].mean(), 2))

# Line plot
st.subheader("Sensor Data Over Time")
sensor_choice = st.selectbox("Choose a variable to visualize:", ['pH', 'Temperature', 'GasFlowRate', 'CH4_Percent'])

fig = px.line(
    data, x='Timestamp', y=sensor_choice,
    color=data['Anomaly'].map({0: 'Normal', 1: 'Anomaly'}),
    color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
    title=f"{sensor_choice} with Anomalies Highlighted"
)
st.plotly_chart(fig, use_container_width=True)

# Table of anomalies
st.subheader("📋 Anomaly Details")
st.dataframe(data[data['Anomaly'] == 1])

# Download
csv = data.to_csv(index=False).encode('utf-8')
st.download_button(
    "📥 Download Data with Anomalies",
    csv,
    "anomaly_detection_results.csv",
    "text/csv",
    key='download-csv'
)

# Real-Time Section
st.title("🔍 Real-Time Anomaly Detection")
placeholder = st.empty()

# Buffer to hold recent data
data_buffer = pd.DataFrame(columns=['Timestamp', 'pH', 'Temperature', 'GasFlowRate', 'CH4_Percent', 'FeedingRate', 'Anomaly', 'Cause'])

while True:
    reading, _ = random_sensor_reading()  # Get simulated sensor data
    is_anomaly = predict_anomaly(model, scaler, reading)
    cause = classifier(classifier_model, scaler, reading) if is_anomaly else None

    row = {
        'Timestamp': datetime.now(),
        'pH': reading[0],
        'Temperature': reading[1],
        'GasFlowRate': reading[2],
        'CH4_Percent': reading[3],
        'FeedingRate': reading[4],
        'Anomaly': is_anomaly,
        'Cause': cause
    }

    data_buffer.loc[len(data_buffer)] = row

    if len(data_buffer) > 50:
        data_buffer = data_buffer.iloc[-50:]

    with placeholder.container():
        st.line_chart(data_buffer.set_index('Timestamp')[['pH', 'Temperature', 'GasFlowRate', 'CH4_Percent', 'FeedingRate']])
        st.markdown("### 📈 Last Reading")
        st.write(row)
        if is_anomaly:
            st.markdown(f"### ⚠️ Detected Anomaly Cause: **{cause}**")
        st.markdown("### 🧾 Recent Anomalies")
        st.dataframe(data_buffer[data_buffer['Anomaly'] == 1])

    time.sleep(2)
