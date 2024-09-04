import os
import pandas as pd

iot_data = []

def process_iot_data(data):
    if data:
        iot_data.append(data)
        df = pd.DataFrame([data])
        if not os.path.isfile('data/real_time_environmental_data.csv'):
            df.to_csv('data/real_time_environmental_data.csv', index=False)
        else:
            df.to_csv('data/real_time_environmental_data.csv', mode='a', header=False, index=False)
