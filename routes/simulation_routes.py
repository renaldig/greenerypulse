# simulation_routes.py
import os
import boto3
from flask import Blueprint, request, jsonify
from services.model_service import invoke_claude_model
from services.map_service import create_map_with_traffic_points, generate_heatmap_based_on_suggestions
from services.rekognition_service import analyze_images_from_s3_folder
from utils.geo_utils import extract_suggestions, extract_locations
from PIL import Image
import base64
import io

simulation_bp = Blueprint('simulation', __name__)

@simulation_bp.route('/simulate', methods=['POST'])
def simulate():
    data = request.json
    prompt = data.get('prompt') + '. Remember that you are an urban planning assistant. Observe the provided markings on the map and provide insights on greenery areas to focus on in the given area. Mention the specific roads or areas in the sawah besar area which you would focus greenery efforts on and make references to the urban markings to supplement this.'
    bucket = data.get('bucket', 'greenerypulseplanning')
    folder = data.get('folder', '')
    terrain_image_key = 'sawah_besar_terrain.png'

    s3_client = boto3.client('s3')
    try:
        s3_response = s3_client.get_object(Bucket=bucket, Key=terrain_image_key)
        image_data = s3_response['Body'].read()
        terrain_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        return jsonify({"error": "Failed to download or process terrain image from S3"}), 500

    image_analysis_results = analyze_images_from_s3_folder(bucket, folder) if folder else None
    ai_recommendations = invoke_claude_model(prompt, image_base64)
    formatted_recommendations = ai_recommendations.replace('\n', '<br>')
    suggestions = extract_suggestions(ai_recommendations)

    bounds = (-6.1650, 106.8200, -6.1450, 106.8500)
    generate_heatmap_based_on_suggestions(terrain_image, suggestions, bounds)

    return jsonify(result=formatted_recommendations, image_analysis=image_analysis_results)

@simulation_bp.route('/traffic-analysis', methods=['POST'])
def traffic_analysis():
    data = request.json
    prompt = "Provide suggestions on optimizing traffic flow in the given area, considering the current traffic density. Focus on the routes and suggest improvements. Make specific references to the street names you make recommendations for and refer to them by Nominatim naming standard (indonesia)."
    bucket = 'greenerypulseplanning'
    traffic_image_key = 'traffic_density_analysis.png'

    s3_client = boto3.client('s3')
    try:
        s3_response = s3_client.get_object(Bucket=bucket, Key=traffic_image_key)
        image_data = base64.b64encode(s3_response['Body'].read()).decode('utf-8')
    except Exception as e:
        return jsonify({"error": "Failed to download traffic density image from S3"}), 500

    try:
        traffic_recommendations_text = invoke_claude_model(prompt, image_data)
    except Exception as e:
        return jsonify({"error": "Failed to invoke Claude model"}), 500

    extracted_locations = extract_locations(traffic_recommendations_text)
    try:
        map_file_path = create_map_with_traffic_points(extracted_locations)
    except Exception as e:
        return jsonify({"error": "Failed to generate traffic routes map"}), 500

    return jsonify(result=traffic_recommendations_text, map_path=map_file_path)
