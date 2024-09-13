import os
from flask import Flask, request, jsonify
import torch
import firebase_admin
from firebase_admin import credentials, firestore
import torchvision.models as models

app = Flask(__name__)

# Initialize Firebase Admin SDK
cred = credentials.Certificate("/home/mich02/Desktop/UMKC_DS_Capstone/firebase_service_account_key.json")
firebase_admin.initialize_app(cred)

# Firestore database reference
db = firestore.client()

# Load the ResNet50 architecture
model = models.resnet50()

# Modify the fully connected layer to match your training setup (2 classes)
model.fc = torch.nn.Linear(in_features=2048, out_features=2)

# Load the saved model weights into the modified architecture
model.load_state_dict(torch.load('/home/mich02/Desktop/UMKC_DS_Capstone/facial_stroke_model.pth'))

# Set the model to evaluation mode
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json()

        # Convert input to torch tensor
        inputs = torch.tensor([data['input']], dtype=torch.float32)

        # Make a prediction
        prediction = model(inputs).item()

        # Save prediction to Firestore
        prediction_data = {
            'prediction': prediction
        }
        db.collection('predictions').add(prediction_data)

        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
