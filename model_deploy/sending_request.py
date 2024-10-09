import requests
import base64

# Open the image file and encode it to base64
with open('path_to_image.jpg', 'rb') as img_file:
    frame_base64 = base64.b64encode(img_file.read()).decode('utf-8')

# Send the request to the Flask API
url = 'http://127.0.0.1:5000/predict'
headers = {'Content-Type': 'application/json'}
data = {'frame': frame_base64}

response = requests.post(url, json=data, headers=headers)

# Print the response
print(response.json())
