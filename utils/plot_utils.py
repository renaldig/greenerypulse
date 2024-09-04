import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def generate_dashboard_image(df, time_frame='1d'):
    """Generate a dashboard image from the data frame."""
    plt.figure(figsize=(16, 10))
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    time_frames = {'30min': 6, '1h': 12, '1d': 288, '1w': 2016, '1m': 8640}
    if time_frame in time_frames:
        df = df.iloc[-time_frames[time_frame]:]

    df.plot(x='timestamp', y='temperature', ax=axes[0, 0], title='Temperature over Time', color='blue', legend=False)
    df.plot(x='timestamp', y='humidity', ax=axes[0, 1], title='Humidity over Time', color='orange', legend=False)
    df.plot(x='timestamp', y='air_quality', ax=axes[1, 0], title='Air Quality over Time', color='green', legend=False)

    if 'noise_level' in df.columns:
        df.plot(x='timestamp', y='noise_level', ax=axes[1, 1], title='Noise Level over Time', color='red', legend=False)
    else:
        axes[1, 1].set_title('Noise Level data not available')
        axes[1, 1].set_visible(False)

    if 'light_intensity' in df.columns:
        df.plot(x='timestamp', y='light_intensity', ax=axes[2, 0], title='Light Intensity over Time', color='purple', legend=False)
    else:
        axes[2, 0].set_title('Light Intensity data not available')
        axes[2, 0].set_visible(False)

    for ax in axes.flatten():
        ax.set_xlabel('Timestamp')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('static/images/dashboard.png')

def generate_predictive_dashboard_image(df):
    """Generate a predictive dashboard image from the data frame."""
    predictions = {feature: perform_predictive_analysis(df, feature) for feature in df.columns if feature != 'timestamp'}

    plt.figure(figsize=(16, 10))
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    time_frames = ['30min', '1h', '1d', '1w', '1m']
    for idx, feature in enumerate(['temperature', 'humidity', 'air_quality', 'noise_level', 'light_intensity']):
        prediction_axis = axes[idx // 2, idx % 2]
        if feature in predictions and isinstance(predictions[feature]['30min'], np.ndarray):
            for time_frame in time_frames:
                if isinstance(predictions[feature][time_frame], np.ndarray):
                    x_future = pd.date_range(start=df['timestamp'].iloc[-1], periods=len(predictions[feature][time_frame]), freq='5T')
                    prediction_axis.plot(x_future, predictions[feature][time_frame], label=f'Prediction {time_frame}')
            prediction_axis.set_title(f'{feature.capitalize()} Predictions')
            prediction_axis.legend()
        else:
            prediction_axis.set_title(f'{feature.capitalize()} Prediction: Not enough data points')
            prediction_axis.set_visible(True)

    for ax in axes.flatten():
        ax.set_xlabel('Timestamp')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('static/images/predictive_dashboard.png')

def perform_predictive_analysis(df, feature):
    """Perform predictive analysis and return predictions."""
    time_intervals = {'30min': 6, '1h': 12, '1d': 288, '1w': 2016, '1m': 8640}
    predictions = {}

    for key, data_points in time_intervals.items():
        if len(df) >= data_points:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[feature].values

            model = LinearRegression()
            model.fit(X, y)

            future_X = np.arange(len(df), len(df) + data_points).reshape(-1, 1)
            y_pred = model.predict(future_X)

            predictions[key] = y_pred
        else:
            predictions[key] = "Not enough data points"

    return predictions
