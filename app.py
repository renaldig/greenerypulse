# app.py
from flask import Flask
from socketio_instance import socketio  # Import socketio from socketio_instance.py
from routes.chat_routes import chat_bp
from routes.simulation_routes import simulation_bp
from routes.iot_routes import iot_bp
from routes.dashboard_routes import dashboard_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'

# Initialize SocketIO with the Flask app
socketio.init_app(app, cors_allowed_origins="*")

# Register Blueprints
app.register_blueprint(chat_bp)
app.register_blueprint(simulation_bp)
app.register_blueprint(iot_bp)
app.register_blueprint(dashboard_bp)

if __name__ == '__main__':
    # Use Flask's built-in server instead of eventlet
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
