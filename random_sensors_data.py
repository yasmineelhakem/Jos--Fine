import random

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
