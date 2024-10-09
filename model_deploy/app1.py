from flask import Flask, request, jsonify
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# Load the stroke detection model from Hugging Face
stroke_model = pipeline('image-classification', model='labteral/groq-stroke-detector-fastc')

def combine_models(image):
    # Make predictions using the stroke model
    return stroke_model(image)

@app.route('/predict', methods=['POST'])
def predict_motion_event():
    try:
        # Get the base64-encoded image from the request
        frame_base64 = request.json['frame']
        
        # Decode the base64 image
        frame_bytes = base64.b64decode(frame_base64)
        
        # Convert bytes into a PIL image (which is compatible with the pipeline)
        image = Image.open(BytesIO(frame_bytes))

        # Call the model for prediction
        prediction = combine_models(image)
        
        # Return the predictions as a JSON response
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Flask runs separately on its own port
