# ♻️ Bio-phine: AI-Powered Anaerobic Digestion of Coffee Grounds for Biogas Production

## Project Overview
Bio-phine is an innovative project developed to address organic waste management and renewable energy generation.  
Our solution focuses on:

- Recovering used coffee grounds to produce biogas (biomethane) through anaerobic digestion
- Integrating real-time monitoring of the reaction environment via IoT sensors
- Applying AI for predictive maintenance and anomaly classification to ensure optimal biogas production

## Software Integration
The AI and software modules include:

### Real-Time Dashboard
- Live forecasting of environmental parameters such as pH, temperature, and CO₂ concentration
- Data visualization and analytics to monitor the digestion process

### Predictive Maintenance
- Anomaly detection using an Isolation Forest model to identify early signs of process disruption
- Immediate signaling of potential issues to prevent digester failure

### Threat Classification
- When an anomaly is detected, an XGBoost Classifier determines the root cause (e.g., acidification, temperature drop)
- Intelligent suggestions for corrective actions are displayed on the dashboard

## Hardware Architecture
- Full conception and assembly of the IoT system using a Raspberry Pi board
- Sensors network measuring pH, temperature, CO₂%, and other critical metrics
- Solidworks design of the anaerobic digester with optimized sensor placement and an integrated mixing mechanism for improved digestion efficiency

## Input Features
- pH_level
- temperature (°C)
- CO₂ concentration (%)
- (Optional) pressure, mixing speed, biogas flow rate

## Output
- anomaly_detected (Yes/No)
- threat_classification (e.g., acidification, cooling issue, sensor fault)

## Dataset
Due to the absence of publicly available datasets for coffee ground anaerobic digestion, a synthetic dataset was generated.  
This synthetic data simulates a wide range of operational scenarios and potential failure modes to train and validate the AI models effectively.

