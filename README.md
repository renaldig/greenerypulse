# GreeneryPulse

GreeneryPulse is an open-source project designed to provide real-time environmental data visualization and analysis. It leverages various AWS services to collect and analyze data, including temperature, humidity, air quality, noise levels, and light intensity. The project includes a Flask-based web application that displays dashboards and offers predictive insights for urban planning and traffic management.

## Features

- **Real-time Environmental Data Dashboard**: Displays updated environmental data using charts.
- **Urban Planning Simulator**: Provides AI-driven insights and recommendations for urban greenery planning.
- **Traffic Analysis**: Analyzes traffic flow and provides suggestions for optimization.
- **Predictive Dashboard**: Predicts future environmental conditions based on historical data.
- **Chat Interface**: Allows users to interact with the system using natural language queries.

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/renaldig/greenerypulse.git
    cd greenerypulse
    ```

2. **Install Dependencies**

    Make sure you have Python installed and then run:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set Up Environment Variables**

    Create a `.env` file in the root directory and add your AWS credentials:

    ```
    aws_access_key_id_3=YOUR_ACCESS_KEY
    aws_secret_access_key_3=YOUR_SECRET_KEY
    bot_id=YOUR_LEX_BOT_ID
    bot_alias=YOUR_LEX_BOT_ALIAS
    ```

4. **Run the Application**

    Start the Flask application using:

    ```bash
    python main.py
    ```

    The application will be available at `http://localhost:5000`.

## Usage

- **Access the Main Dashboard**: Navigate to `http://localhost:5000` to view the real-time environmental data and interact with the dashboards.
- **Urban Planning Simulator**: Use the prompt box to input queries related to urban planning and receive AI-generated insights.
- **Traffic Analysis**: Analyze traffic data and get optimization suggestions directly on the dashboard.

## Project Structure

- **`app.py`**: The main entry point for the Flask application, setting up the server and registering blueprints.
- **`routes/`**: Contains the route handlers for different functionalities.
  - **`iot_routes.py`**: Handles routes related to IoT data ingestion and processing.
  - **`dashboard_routes.py`**: Manages routes for rendering dashboards and handling related socket events.
  - **`simulation_routes.py`**: Contains routes for urban planning simulation and traffic analysis.
- **`services/`**: Contains service modules that encapsulate business logic.
  - **`data_service.py`**: Handles data processing, dashboard generation, and predictive analytics.
  - **`lex_service.py`**: Interacts with AWS Lex for chat functionalities.
  - **`rekognition_service.py`**: Handles image recognition tasks using AWS Rekognition.
  - **`model_service.py`**: Interfaces with AI models for generating insights.
  - **`map_service.py`**: Manages map creation and visualization functionalities.
- **`utils/`**: Contains utility modules for common helper functions.
  - **`geo_utils.py`**: Provides geographical data processing utilities.
  - **`file_utils.py`**: Handles file-related utilities.
  - **`plot_utils.py`**: Contains utilities for plotting and data visualization.
- **`static/`**: Contains CSS, JavaScript files, and static images.
- **`templates/`**: Contains HTML templates for the web pages.
- **`data/`**: Stores CSV files for environmental data.
- **`certs/`**: Placeholder for SSL certificates.


## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
