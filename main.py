import re
import cv2
import os
import json
import random
import requests
import imgkit
import time
import base64
from imagerecognizer import analyze_images_from_s3_folder
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import uuid
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from PIL import Image, ImageDraw
import io
import folium

# Set Matplotlib backend to Agg
matplotlib.use('Agg')

# Initialize the Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
geocode_cache = {}

session = boto3.Session(
    aws_access_key_id=os.getenv('aws_access_key_id_3'),
    aws_secret_access_key=os.getenv('aws_secret_access_key_3'),
    region_name='us-west-2'
)

# Initialize AWS clients
lex_client = session.client('lexv2-runtime', region_name='us-west-2')
bedrock_client = session.client('bedrock-runtime', region_name='us-west-2')
rekognition_client = session.client('rekognition')
iot_client = session.client('iot-data', region_name='us-west-2')
dynamodb_client = session.client('dynamodb', region_name='us-west-2')

# In-memory storage for IoT data
iot_data = []

# Function to interact with Lex bot
def send_to_lex_bot(bot_id, bot_alias_id, locale_id, user_id, text):
    response = lex_client.recognize_text(
        botId=bot_id,
        botAliasId=bot_alias_id,
        localeId=locale_id,
        sessionId=user_id,
        text=text
    )

    print("Lex Response:", response)

    # Check for interpretations first
    if 'interpretations' in response and response['interpretations']:
        print("Hello")
        for interpretation in response['interpretations']:
            if 'intent' in interpretation and interpretation['intent']['name'] == 'ProvideFeedback':
                intent_state = interpretation['intent']['state']
                slots = interpretation['intent'].get('slots', {})
                feedback_slot = slots.get('FeedbackText')
                
                # Only proceed if the intent state is 'ReadyForFulfillment'
                if intent_state == 'ReadyForFulfillment' and feedback_slot and feedback_slot.get('value'):
                    feedback_text = feedback_slot['value'].get('interpretedValue')
                    print(f"The feedback is {feedback_text}")
                    if feedback_text:
                        # Insert feedback into DynamoDB
                        insert_feedback_to_dynamodb(feedback_text)
                        # Return success message
                        return "Thank you for your feedback!"
                elif feedback_slot is None or feedback_slot.get('value') is None:
                    return "Please provide your feedback."
                else:
                    return "Processing your feedback, please wait."
            elif 'intent' in interpretation and interpretation['intent']['name'] == 'FallbackIntent':
                return handle_fallback(text)

    # Only check for messages if no relevant intent found
    if 'messages' in response and response['messages']:
        print("Hola")
        return response['messages'][0]['content']
    else:
        return "No valid response from the bot."

    
@app.route('/chat', methods=['POST'])
def chat():
    # Get the message sent by the user
    data = request.json
    user_message = data.get('text', '')

    # You should have the necessary logic to handle the user message here
    # For example, you could send the message to your Lex bot
    bot_response = send_to_lex_bot(bot_id=os.getenv('bot_id'), bot_alias_id=os.getenv('bot_alias'), locale_id='en_US', user_id='greenerypulsebot', text=user_message)
    
    # Return the bot response to the client
    return jsonify(response=bot_response)
    
def insert_feedback_to_dynamodb(feedback_text):
    # Generate a unique numeric ID
    feedback_id = random.randint(1, 1000000)  # Ensure this range fits within your ID requirements
    
    # Current timestamp
    timestamp = int(time.time())
    
    print(f"Attempting to insert feedback into DynamoDB: FeedbackText={feedback_text}, FeedbackId={feedback_id}, Timestamp={timestamp}")
    
    # Insert the item into the DynamoDB table
    try:
        response = dynamodb_client.put_item(
            TableName='GreeneryPulseFeedbackTable',
            Item={
                'id': {'N': str(feedback_id)},  # Convert feedback_id to string
                'Timestamp': {'N': str(timestamp)},  # Convert timestamp to string
                'FeedbackText': {'S': feedback_text}
            }
        )
        print(f"Successfully inserted feedback with ID: {feedback_id}, DynamoDB response: {response}")
    except Exception as e:
        print(f"Error inserting feedback into DynamoDB: {e}")
    
def handle_fallback(user_text):
    """Handle fallback by calling Bedrock and returning a response."""
    try:
        # Call Bedrock with the fallback message
        prompt = user_text + "Answer in the context of an urban planner assistant."
        result = invoke_claude_model(prompt)
        return result
    except Exception as e:
        print("Error in fallback:", e)
        return "I'm having trouble understanding. Can you please rephrase?"

# Function to invoke Claude model for urban planning simulation
def invoke_claude_model(prompt, image_data=None):
    payload = {
        "max_tokens": 1024,
        "system": "You are an urban planning assistant. Act like it and respond based on the prompt provided.",  # Replace with relevant system prompt
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "anthropic_version": "bedrock-2023-05-31"
    }

    if image_data:
        payload["messages"][0]["content"].append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            }
        )

    try:
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(payload),
            contentType="application/json"
        )

        raw_response = response['body'].read().decode('utf-8')
        print(f"Raw Response: {raw_response}")

        if not raw_response.strip():
            raise ValueError("Received an empty response from the Claude model.")

        try:
            response_body = json.loads(raw_response)
            print("Response Body:", response_body)

            # Parse the text content directly from the content field
            if 'content' in response_body:
                text_content = response_body['content'][0].get('text', '')
                return text_content
            else:
                raise KeyError("Unexpected response structure: 'content' field not found.")
        except json.JSONDecodeError as e:
            raise ValueError("Failed to parse JSON response. The response might be empty or improperly formatted.") from e

    except Exception as e:
        print(f"Error in invoke_claude_model: {e}")
        raise


# Function to analyze images using Rekognition
def analyze_image_from_s3(bucket, key):
    response = rekognition_client.detect_labels(
        Image={'S3Object': {'Bucket': bucket, 'Name': key}},
        MaxLabels=10
    )
    return response['Labels']

def create_map_with_traffic_points(locations):
    """Function to create a map with traffic points."""
    # Define the initial location and zoom level
    sawah_besar_center = [-6.1550, 106.8350]  # Center of Sawah Besar
    m = folium.Map(location=sawah_besar_center, zoom_start=15)

    # Add traffic points
    for location in locations:
        geocoded_coords = geocode_location(location)
        if geocoded_coords:
            lat, lon = geocoded_coords
            folium.Marker([lat, lon], popup=location).add_to(m)
        else:
            print(f"Geocoding failed for {location}")

    # Save the map as an HTML file
    map_file_path = 'static/images/traffic_routes_map.html'
    m.save(map_file_path)
    print(f"Map with traffic points saved as '{map_file_path}'.")

    return map_file_path

def lat_lon_to_image_coords(lat, lon, img_width, img_height, bounds):
    min_lat, min_lon, max_lat, max_lon = bounds
    
    x = int((lon - min_lon) / (max_lon - min_lon) * img_width)
    y = int((max_lat - lat) / (max_lat - min_lat) * img_height)
    
    # Ensure coordinates are within bounds
    x = max(0, min(img_width - 1, x))
    y = max(0, min(img_height - 1, y))
    
    return x, y

def plot_location_on_map(route_map_data, lat, lon, img_width, img_height, bounds, offset=0):
    x, y = lat_lon_to_image_coords(lat, lon, img_width, img_height, bounds)
    print(f"Plotting location at: ({x}, {y}) with offset: {offset}")
    
    # Ensure coordinates are within bounds
    if 0 <= x < img_width and 0 <= y < img_height:
        # Apply offset to avoid overlapping points
        x = min(img_width - 1, max(0, x + offset))
        y = min(img_height - 1, max(0, y + offset))
        
        # Draw a filled circle instead of a single pixel
        cv2.circle(route_map_data, (x, y), 5, (255, 0, 0), -1)  # Red color for visibility
    else:
        print(f"Coordinates out of bounds for location: ({x}, {y})")

def geocode_location(location_name):
    """Function to geocode a location using the Nominatim API with caching and rate limiting."""
    # Check cache first
    if location_name in geocode_cache:
        print(f"Using cached result for {location_name}")
        return geocode_cache[location_name]
    
    clean_location_name = location_name.replace("Jalan ", "")
    base_url = "https://nominatim.openstreetmap.org/search"
    
    # List of queries to try
    queries = [
        f"{clean_location_name}, Jakarta, Indonesia",
        f"{clean_location_name}",
        f"{clean_location_name.split(',')[0]}",
    ]
    
    headers = {
        'User-Agent': 'MyTrafficOptimizationApp/1.0 (theradprepx@gmail.com)'  # Replace with your app info
    }
    
    for query in queries:
        params = {
            "q": query,
            "format": "json",
            "limit": 5,
            "countrycodes": "ID"  # Restrict results to Indonesia
        }
        try:
            response = requests.get(base_url, params=params, headers=headers)
            print(f"Geocoding query: {query}")  # Debugging statement
            print(f"Response status code: {response.status_code}")  # Debugging statement
            if response.status_code == 200:
                results = response.json()
                print(f"Results: {results}")  # Debugging statement
                if results:
                    for result in results:
                        if "Jakarta" in result.get('display_name', ''):
                            lat_lon = (float(result['lat']), float(result['lon']))
                            geocode_cache[location_name] = lat_lon  # Cache the result
                            print(f"Found location: {result['display_name']}")  # Debugging statement
                            return lat_lon
            else:
                print(f"Geocoding failed for location: {query} with status code {response.status_code}")
        except requests.RequestException as e:
            print(f"Request failed: {e}")

        time.sleep(1)  # Ensure we don't exceed the 1 request per second limit

    print(f"No suitable match found for location: {location_name}")
    return None

def extract_locations(text):
    # Define a regex pattern to extract streets prefixed with 'Jalan' or 'Jl.'
    street_pattern = re.compile(r'\b[Jj]alan(?:\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)')
    non_location_keywords = [
    "Based", "Here", "This", "To", "Additionally", "Similar", "An", 
    "The", "Implementing", "Intersections", "Reconfiguring", 
    "Mass Transit", "Exploring", "Let", "Ensuring", "Considering", 
    "For", "Long", "Widening", "Improving", "Intersection", "Optimizing", 
    "Enhancing", "Public", "Encouraging"
    ]

    # Find all matches
    locations = street_pattern.findall(text)
    
    # Filter out any unwanted matches
    filtered_locations = []
    for loc in locations:
        clean_loc = loc.strip()
        # Add additional cleaning logic if needed, for example:
        if clean_loc.lower() not in non_location_keywords:
            filtered_locations.append(clean_loc)
    
    print("Filtered Locations: ", filtered_locations)
    return filtered_locations

def extract_location(line):
    """Extracts location from a line of text using regex for street names."""
    match = re.search(r'\bJalan\s[A-Z][a-z]*(?:\s[A-Z][a-z]*)*', line)
    if match:
        return match.group()
    return None

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    prompt = data.get('prompt') + '. Remember that you are an urban planning assistant. Observe the provided markings on the map and provide insights on greenery areas to focus on in the given area. Mention the specific roads or areas in the sawah besar area which you would focus greenery efforts on and make references to the urban markings to supplement this.'

    bucket = data.get('bucket', 'greenerypulseplanning')
    folder = data.get('folder', '')
    terrain_image_key = 'sawah_besar_terrain.png'

    s3_client = session.client('s3')
    try:
        s3_response = s3_client.get_object(Bucket=bucket, Key=terrain_image_key)
        
        # Verify content type
        content_type = s3_response.get('ContentType', '')
        if 'image' not in content_type:
            raise ValueError(f"Expected image content type, but got {content_type}")
        
        image_data = s3_response['Body'].read()
        terrain_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        
        # Convert image data to base64 for Claude model
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
    except Exception as e:
        return jsonify({"error": "Failed to download or process terrain image from S3"}), 500

    image_analysis_results = None
    if folder:
        image_analysis_results = analyze_images_from_s3_folder(bucket, folder)
    
    ai_recommendations = invoke_claude_model(prompt, image_base64)
    
    # Ensure the AI recommendations are formatted for readability
    formatted_recommendations = ai_recommendations.replace('\n', '<br>')  # Replace newline characters with HTML line breaks

    suggestions = extract_suggestions(ai_recommendations)

    bounds = (-6.1650, 106.8200, -6.1450, 106.8500)
    generate_heatmap_based_on_suggestions(terrain_image, suggestions, bounds)

    return jsonify(result=formatted_recommendations, image_analysis=image_analysis_results)

def extract_suggestions(ai_recommendations):
    """Extracts structured suggestions from the AI's textual recommendations."""
    suggestions = []
    lines = ai_recommendations.split("\n")
    for line in lines:
        location = extract_location(line)
        if location:
            action = determine_action(line)
            suggestions.append({'location': location, 'action': action})
    return suggestions

# Function to determine action from AI response
def determine_action(line):
    """Determines the action based on keywords in the AI response line."""
    if "greenery" in line.lower() or "tree" in line.lower() or "vegetation" in line.lower():
        return 'greenery_focus'
    elif "traffic" in line.lower() or "congestion" in line.lower():
        return 'high_traffic'
    else:
        return 'neutral'


def generate_heatmap_based_on_suggestions(terrain_image, suggestions, bounds):
    """Generate a heatmap image based on AI suggestions."""
    # Define the color map
    colormap = {
        'high_traffic': (255, 0, 0, 128),  # Red with transparency
        'greenery_focus': (0, 255, 0, 128),  # Green with transparency
        'neutral': (255, 255, 0, 128)  # Yellow with transparency for visibility
    }

    # Convert the image to a numpy array
    image_array = np.array(terrain_image)

    # Create a new heatmap layer
    heatmap = Image.new('RGBA', terrain_image.size)

    # Draw on the heatmap based on suggestions
    draw = ImageDraw.Draw(heatmap)

    # Iterate through the suggestions and draw corresponding areas on the heatmap
    for suggestion in suggestions:
        location = suggestion.get('location')
        action = suggestion.get('action')

        # Convert location to image coordinates
        coords = geocode_location(location)
        if coords:
            x, y = lat_lon_to_image_coords(coords[0], coords[1], image_array.shape[1], image_array.shape[0], bounds)
            color = colormap.get(action, (255, 255, 255, 255))  # Use fully opaque white if action not found
            # Increase the size of the rectangles (e.g., 20x20 pixels instead of 10x10)
            draw.rectangle([x-100, y-100, x+100, y+100], fill=color)
            print(f"Drawing {action} at ({x}, {y}) with color {color} and larger size")  # Debugging statement

    # Combine the heatmap with the original image
    combined = Image.alpha_composite(terrain_image.convert('RGBA'), heatmap)

    # Save the combined image
    combined.save('static/images/urban_heatmap.png')
    print("Heatmap generated based on suggestions.")

def analyze_map_and_generate_traffic_routes(bucket, traffic_image_key, traffic_recommendations_text):
    s3_client = session.client('s3')

    # Download the traffic density map image from S3 for analysis purposes
    try:
        s3_response = s3_client.get_object(Bucket=bucket, Key=traffic_image_key)
        image_data = s3_response['Body'].read()
        base_map_image = Image.open(io.BytesIO(image_data)).convert("RGBA")  # Keep for Claude analysis
        base_map_array = np.array(base_map_image)
    except Exception as e:
        print(f"Error downloading traffic image: {e}")
        return jsonify({"error": f"Failed to download traffic density image from S3: {e}"}), 500

    # Extract locations from Sonnet's response
    extracted_locations = extract_locations(traffic_recommendations_text)
    print("Extracted Locations:", extracted_locations)

    # Create a Folium map centered on Sawah Besar
    sawah_besar_center = [(-6.1650 + -6.1450) / 2, (106.8200 + 106.8500) / 2]
    folium_map = folium.Map(location=sawah_besar_center, zoom_start=15)

    # Plotting the points directly onto the Folium map
    for location in extracted_locations:
        geocoded_coords = geocode_location(location)
        if geocoded_coords:
            lat, lon = geocoded_coords
            folium.Marker([lat, lon], popup=location).add_to(folium_map)
        else:
            print(f"Geocoding failed for {location}")

    # Save the map as an HTML file
    traffic_map_path = 'static/images/traffic_routes_map.html'
    folium_map.save(traffic_map_path)
    print(f"Traffic Routes Map saved as '{traffic_map_path}'.")

    # Also save as PNG for display
    png_output = 'static/images/traffic_routes_map.png'
    imgkit.from_file(traffic_map_path, png_output)
    print(f"Traffic Routes Map PNG saved as '{png_output}'.")

# Function to receive data from IoT Core and update the dashboard
@app.route('/receive-iot-data', methods=['POST'])
def receive_iot_data():
    print("Received request to /receive-iot-data")
    print("Headers:", request.headers)
    print("Body:", request.get_data(as_text=True))
    data = request.json
    if data and "confirmationUrl" in data:
        # Handle the confirmation request
        print("Confirmation request received:", data)
        return jsonify({"message": "Endpoint confirmed"}), 200
    else:
        print("Received data:", data)  # Debugging: print received data
        iot_data.append(data)
        
        # Append new data to the CSV instead of overwriting
        df = pd.DataFrame([data])
        if not os.path.isfile('data/real_time_environmental_data.csv'):
            df.to_csv('data/real_time_environmental_data.csv', index=False)
        else:
            df.to_csv('data/real_time_environmental_data.csv', mode='a', header=False, index=False)
        
        generate_dashboard_image(df)
        socketio.emit('update_dashboard')
        return jsonify(status="Data received")

    
# Function to perform predictive analysis
def perform_predictive_analysis(df, feature):
    """Perform predictive analysis and return predictions for future time frames."""
    time_intervals = {'30min': 6, '1h': 12, '1d': 288, '1w': 2016, '1m': 8640}  # Data points needed for predictions
    predictions = {}
    
    for key, data_points in time_intervals.items():
        if len(df) >= data_points:
            # Prepare the data for prediction
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[feature].values
            
            # Train the linear regression model
            model = LinearRegression()
            model.fit(X, y)
            
            # Predict future values
            future_X = np.arange(len(df), len(df) + data_points).reshape(-1, 1)
            y_pred = model.predict(future_X)
            
            predictions[key] = y_pred
        else:
            predictions[key] = "Not enough data points"
    
    return predictions

# Route for the main page
@app.route('/')
def index():
    # Directly render the dashboard on the index page
    return render_template('index.html')

@app.route('/traffic-analysis', methods=['POST'])
@app.route('/traffic-analysis', methods=['POST'])
def traffic_analysis():
    try:
        print("Traffic analysis started...")  # Debugging statement
        data = request.json
        prompt = "Provide suggestions on optimizing traffic flow in the given area, considering the current traffic density. Focus on the routes and suggest improvements. Make specific references to the street names you make recommendations for and refer to them by Nominatim naming standard (indonesia)."
        
        # Fixed bucket and image key for the traffic density map
        bucket = 'greenerypulseplanning'
        traffic_image_key = 'traffic_density_analysis.png'

        # Step 1: Download the traffic density image from S3 and convert it to base64
        s3_client = session.client('s3')
        try:
            s3_response = s3_client.get_object(Bucket=bucket, Key=traffic_image_key)
            image_data = base64.b64encode(s3_response['Body'].read()).decode('utf-8')
            print("Image downloaded and encoded.")  # Debugging statement
        except Exception as e:
            print(f"Error downloading image: {e}")
            return jsonify({"error": "Failed to download traffic density image from S3"}), 500

        # Step 2: Invoke the Claude model with the prompt and image
        try:
            traffic_recommendations_text = invoke_claude_model(prompt, image_data)
            print("Claude model invoked.")  # Debugging statement
        except Exception as e:
            print(f"Error invoking Claude model: {e}")
            return jsonify({"error": "Failed to invoke Claude model"}), 500

        # Extracted locations from Claude model's response
        extracted_locations = extract_locations(traffic_recommendations_text)
        print(f"Extracted Locations: {extracted_locations}")

        # Step 4: Generate map with traffic points based on AI recommendations
        try:
            map_file_path = create_map_with_traffic_points(extracted_locations)
            print("Traffic routes map generated.")  # Debugging statement
        except Exception as e:
            print(f"Error generating traffic routes map: {e}")
            return jsonify({"error": "Failed to generate traffic routes map"}), 500

        # Step 5: Return both Claude model results and the HTML map path
        return jsonify(result=traffic_recommendations_text, map_path="static/images/traffic_routes_map.html")
    
    except Exception as e:
        print(f"General error in traffic analysis: {e}")
        return jsonify({"error": str(e)}), 500

def generate_dashboard_image(df, time_frame='1d'):
    if 'timestamp' not in df.columns:
        print("Error: 'timestamp' column not found in DataFrame")
        return

    plt.figure(figsize=(16, 10))
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    # Time frame filtering logic
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
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('static/images/dashboard.png')
    print("Dashboard image saved.")

def generate_predictive_dashboard_image(df):
    if 'timestamp' not in df.columns:
        print("Error: 'timestamp' column not found in DataFrame")
        return

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
        ax.set_ylabel('')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('static/images/predictive_dashboard.png')
    print("Predictive Dashboard image saved.")

@app.route('/generate-insights', methods=['POST'])
def generate_insights():
    data = request.json
    # Load the environmental data from the CSV or in-memory storage
    df = pd.read_csv('data/real_time_environmental_data.csv')
    
    # Generate a summary or description of the current environmental data
    summary = df.describe().to_dict()

    # Construct a prompt for the Sonnet model using the summary of the data
    prompt = (
        "Given the following environmental data summary: \n"
        f"{json.dumps(summary, indent=2)}\n"
        "Please provide insights and recommendations on how to achieve better energy or air quality efficiency."
    )

    # Call the Claude model for insights
    insights = invoke_claude_model(prompt)

    return jsonify(insights=insights)

# Real-time updates with WebSockets
@socketio.on('connect')
def handle_connect():
    print("Client connected")
    emit('update_dashboard')

@socketio.on('update_time_frame')
def handle_update_time_frame(data):
    time_frame = data.get('time_frame')
    dashboard_type = data.get('type')
    
    print(f"Updating {dashboard_type} dashboard with time frame: {time_frame}")

    if os.path.exists('data/real_time_environmental_data.csv'):
        df = pd.read_csv('data/real_time_environmental_data.csv')
        if dashboard_type == 'normal':
            generate_dashboard_image(df, time_frame)
        elif dashboard_type == 'predictive':
            generate_predictive_dashboard_image(df)

    emit('update_dashboard')

@socketio.on('request_update')
def handle_request_update():
    print("Update request received")
    if os.path.exists('data/real_time_environmental_data.csv'):
        df = pd.read_csv('data/real_time_environmental_data.csv')
        print("Data read from CSV:")
        print(df.head()) 
        generate_dashboard_image(df)
        generate_predictive_dashboard_image(df)
    emit('update_dashboard')

if __name__ == '__main__':
    socketio.run(app, debug=True)