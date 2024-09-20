from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
from werkzeug.utils import secure_filename
from pymongo import MongoClient

# Initialize the Flask app
app = Flask(__name__, template_folder="../frontend/templates")

# Path where images will be uploaded
UPLOAD_FOLDER = '../uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["UMKC_DS_Capstone"]
collection = db["stroke_detection_data"]

# Define the ResNet50 model architecture with a modified fully connected layer (2 output classes)
model = models.resnet50(pretrained=False)  # Do not load pretrained ImageNet weights
model.fc = torch.nn.Linear(in_features=2048, out_features=2)  # Modify the FC layer for 2 classes

# Load the state dictionary (weights) into the model
model.load_state_dict(torch.load('facial_stroke_model.pth'))

# Set the model to evaluation mode
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to match model's input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet values
])

# Route to display the home page
@app.route('/', methods=['GET'])
def hello_world():
    return render_template("home.hbs")

# Route to handle image upload and classification
@app.route('/predict', methods=['POST'])
def upload_image():
    if 'imagefile' not in request.files:
        return "No file part"
    
    imagefile = request.files['imagefile']
    
    if imagefile.filename == '':
        return "No selected file"
    
    if imagefile:
        # Ensure filename is secure
        filename = secure_filename(imagefile.filename)
        
        # Save the file to the upload folder
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        imagefile.save(filepath)

        # Open the image for classification
        image = Image.open(filepath).convert('RGB')
        
        # Apply transformations to the image
        image = transform(image).unsqueeze(0)  # Add batch dimension

        # Pass the image through the model
        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)  # Get the predicted class
            
        # Define the labels (example: 0 = no stroke, 1 = stroke)
        labels = ['No Stroke', 'Stroke']
        result = labels[predicted.item()]  # Convert prediction to label
        
        # Save the image path and result to the database
        collection.insert_one({"image_path": filepath, "result": result})
        
        # Return the result to the frontend
        return render_template("result.hbs", result=result)
    
if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(port=3002, debug=True)
