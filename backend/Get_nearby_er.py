import requests

# Google Places API call to search for nearby emergency rooms
def find_nearby_ers(user_location):
    latitude, longitude = user_location  # User's location
    radius = 5000  # 5 km radius
    api_key = "GOOGLE_API"  # Replace with your actual Google API key

    # Construct the Places API URL
    url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        'location': f'{latitude},{longitude}',
        'radius': radius,
        'type': 'hospital',
        'keyword': 'emergency',
        'key': api_key
    }

    # Send the request to the Google Places API
    response = requests.get(url, params=params)

    # Parse the JSON response
    if response.status_code == 200:
        places_data = response.json()

        # Extract relevant data from the API response
        if 'results' in places_data:
            er_list = []
            for place in places_data['results']:
                er_list.append({
                    'name': place.get('name'),
                    'address': place.get('vicinity'),
                    'rating': place.get('rating'),
                    'user_ratings_total': place.get('user_ratings_total'),
                    'location': place['geometry']['location']
                })
            return er_list
        else:
            return []
    else:
        print(f"Error: {response.status_code}")
        return []

# Example of calling the function
user_location = (39.0997, -94.5786)  # Example: Kansas City, MO (latitude, longitude)
ers = find_nearby_ers(user_location)
for er in ers:
    print(er)
