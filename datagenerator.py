import pandas as pd
import numpy as np
import time

def generate_mock_data():
    timestamps = pd.date_range(start="2023-01-01", periods=100, freq='H')
    data = {
        'timestamp': timestamps,
        'temperature': np.random.normal(loc=25, scale=5, size=100),
        'humidity': np.random.uniform(low=30, high=80, size=100),
        'air_quality': np.random.uniform(low=0, high=100, size=100)
    }
    df = pd.DataFrame(data)
    return df

def main():
    df = generate_mock_data()
    df.to_csv('mock_environmental_data.csv', index=False)
    print("Mock environmental data generated and saved to 'mock_environmental_data.csv'.")

if __name__ == "__main__":
    main()
