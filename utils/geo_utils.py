import re
import requests
import time

geocode_cache = {}

def geocode_location(location_name):
    if location_name in geocode_cache:
        return geocode_cache[location_name]

    clean_location_name = location_name.replace("Jalan ", "")
    base_url = "https://nominatim.openstreetmap.org/search"

    queries = [
        f"{clean_location_name}, Jakarta, Indonesia",
        f"{clean_location_name}",
        f"{clean_location_name.split(',')[0]}",
    ]

    headers = {
        'User-Agent': 'MyTrafficOptimizationApp/1.0 (theradprepx@gmail.com)'
    }

    for query in queries:
        params = {
            "q": query,
            "format": "json",
            "limit": 5,
            "countrycodes": "ID"
        }
        try:
            response = requests.get(base_url, params=params, headers=headers)
            if response.status_code == 200:
                results = response.json()
                if results:
                    for result in results:
                        if "Jakarta" in result.get('display_name', ''):
                            lat_lon = (float(result['lat']), float(result['lon']))
                            geocode_cache[location_name] = lat_lon
                            return lat_lon
        except requests.RequestException as e:
            print(f"Request failed: {e}")

        time.sleep(1)

    return None

def lat_lon_to_image_coords(lat, lon, img_width, img_height, bounds):
    min_lat, min_lon, max_lat, max_lon = bounds

    x = int((lon - min_lon) / (max_lon - min_lon) * img_width)
    y = int((max_lat - lat) / (max_lat - min_lat) * img_height)

    x = max(0, min(img_width - 1, x))
    y = max(0, min(img_height - 1, y))

    return x, y

def extract_suggestions(ai_recommendations):
    suggestions = []
    lines = ai_recommendations.split("\n")
    for line in lines:
        location = extract_location(line)
        if location:
            action = determine_action(line)
            suggestions.append({'location': location, 'action': action})
    return suggestions

def extract_location(line):
    match = re.search(r'\bJalan\s[A-Z][a-z]*(?:\s[A-Z][a-z]*)*', line)
    if match:
        return match.group()
    return None

def determine_action(line):
    if "greenery" in line.lower() or "tree" in line.lower() or "vegetation" in line.lower():
        return 'greenery_focus'
    elif "traffic" in line.lower() or "congestion" in line.lower():
        return 'high_traffic'
    else:
        return 'neutral'

def extract_locations(text):
    """Extract locations from the text using regex for street names."""
    street_pattern = re.compile(r'\b[Jj]alan(?:\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)')
    non_location_keywords = [
        "Based", "Here", "This", "To", "Additionally", "Similar", "An", 
        "The", "Implementing", "Intersections", "Reconfiguring", 
        "Mass Transit", "Exploring", "Let", "Ensuring", "Considering", 
        "For", "Long", "Widening", "Improving", "Intersection", "Optimizing", 
        "Enhancing", "Public", "Encouraging"
    ]

    locations = street_pattern.findall(text)
    filtered_locations = [loc.strip() for loc in locations if loc.strip().lower() not in non_location_keywords]
    return filtered_locations
