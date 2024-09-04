import os
import pandas as pd
import json
from flask import Blueprint, request, jsonify, render_template
from services.data_service import generate_dashboard_image, generate_predictive_dashboard_image
from services.model_service import invoke_claude_model
from socketio_instance import socketio  # Use the socketio instance from socketio_instance.py

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    return render_template('index.html')

@dashboard_bp.route('/generate-insights', methods=['POST'])
def generate_insights():
    if os.path.exists('data/real_time_environmental_data.csv'):
        df = pd.read_csv('data/real_time_environmental_data.csv')
        summary = df.describe().to_dict()

        prompt = (
            "Given the following environmental data summary: \n"
            f"{json.dumps(summary, indent=2)}\n"
            "Please provide insights and recommendations on how to achieve better energy or air quality efficiency."
        )

        insights = invoke_claude_model(prompt)
        return jsonify(insights=insights)
    else:
        return jsonify(error="Data file not found."), 404

@socketio.on('connect')
def handle_connect():
    socketio.emit('update_dashboard')  # Ensure proper usage with socketio.emit

@socketio.on('update_time_frame')
def handle_update_time_frame(data):
    time_frame = data.get('time_frame')
    dashboard_type = data.get('type')

    if os.path.exists('data/real_time_environmental_data.csv'):
        df = pd.read_csv('data/real_time_environmental_data.csv')
        if dashboard_type == 'normal':
            generate_dashboard_image(df, time_frame)
        elif dashboard_type == 'predictive':
            generate_predictive_dashboard_image(df)
        socketio.emit('update_dashboard')  # Ensure proper usage with socketio.emit
    else:
        socketio.emit('error', {'message': 'Data file not found.'})

@socketio.on('request_update')
def handle_request_update():
    if os.path.exists('data/real_time_environmental_data.csv'):
        df = pd.read_csv('data/real_time_environmental_data.csv')
        generate_dashboard_image(df)
        generate_predictive_dashboard_image(df)
        socketio.emit('update_dashboard')  # Ensure proper usage with socketio.emit
    else:
        socketio.emit('error', {'message': 'Data file not found.'})
