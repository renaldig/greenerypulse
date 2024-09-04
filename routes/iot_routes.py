from flask import Blueprint, request, jsonify
from services.iot_service import process_iot_data
from socketio_instance import socketio  # Import socketio instance from socketio_instance.py

iot_bp = Blueprint('iot', __name__)

@iot_bp.route('/receive-iot-data', methods=['POST'])
def receive_iot_data():
    data = request.json
    if data and "confirmationUrl" in data:
        return jsonify({"message": "Endpoint confirmed"}), 200
    else:
        process_iot_data(data)
        # Emit the event to all connected clients
        socketio.emit('update_dashboard', to='/')  # Use 'to' instead of 'broadcast'
        return jsonify(status="Data received")
