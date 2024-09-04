import folium
from PIL import Image, ImageDraw
import numpy as np
from utils.geo_utils import geocode_location, lat_lon_to_image_coords

def create_map_with_traffic_points(locations):
    sawah_besar_center = [-6.1550, 106.8350]
    m = folium.Map(location=sawah_besar_center, zoom_start=15)

    for location in locations:
        geocoded_coords = geocode_location(location)
        if geocoded_coords:
            lat, lon = geocoded_coords
            folium.Marker([lat, lon], popup=location).add_to(m)
        else:
            print(f"Geocoding failed for {location}")

    map_file_path = 'static/images/traffic_routes_map.html'
    m.save(map_file_path)
    return map_file_path

def generate_heatmap_based_on_suggestions(terrain_image, suggestions, bounds):
    colormap = {
        'high_traffic': (255, 0, 0, 128),
        'greenery_focus': (0, 255, 0, 128),
        'neutral': (255, 255, 0, 128)
    }

    image_array = np.array(terrain_image)
    heatmap = Image.new('RGBA', terrain_image.size)
    draw = ImageDraw.Draw(heatmap)

    for suggestion in suggestions:
        location = suggestion.get('location')
        action = suggestion.get('action')
        coords = geocode_location(location)

        if coords:
            x, y = lat_lon_to_image_coords(coords[0], coords[1], image_array.shape[1], image_array.shape[0], bounds)
            color = colormap.get(action, (255, 255, 255, 255))
            draw.rectangle([x-100, y-100, x+100, y+100], fill=color)

    combined = Image.alpha_composite(terrain_image.convert('RGBA'), heatmap)
    combined.save('static/images/urban_heatmap.png')
