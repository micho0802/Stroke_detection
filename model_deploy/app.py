from flask import Flask, request, jsonify
import os
import torch
from torchvision import models, transforms
from PIL import Image
import joblib  # For loading the RandomForest model
from werkzeug.utils import secure_filename
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Path where the images will be uploaded
UPLOAD_FOLDER = '../uploaded_images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the ResNet50 model with a modified fully connected layer (2 output classes)
facial_model = models.resnet50(weights=None)
facial_model.fc = torch.nn.Linear(in_features=2048, out_features=2)
facial_model.load_state_dict(torch.load('/home/mich02/Desktop/UMKC_DS_Capstone/model_pretrained/facial_stroke_model.pth', weights_only=True)) #Pretrain model can be found in https://huggingface.co/micho02/Face_stroke_detection/tree/main
facial_model.eval()

# Load the RandomForest model (pretrained)
gb_model = joblib.load('/home/mich02/Desktop/UMKC_DS_Capstone/model_pretrained/medical_history_model.pth')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return jsonify({'error': 'No selected image file'}), 400

    # Save the image to the upload folder
    filename = secure_filename(imagefile.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    try:
        imagefile.save(filepath)
    except Exception as e:
        return jsonify({'error': f'Error saving image file: {str(e)}'}), 500

    # Get medical history data from the form
    try:
        gender = request.form.get('gender')
        age = request.form.get('age')
        hypertension = request.form.get('hypertension')
        heart_disease = request.form.get('heart_disease')
        ever_married = request.form.get('ever_married')
        work_type = request.form.get('work_type')
        residence_type = request.form.get('residence_type')
        avg_glucose_level = request.form.get('avg_glucose_level')
        bmi = request.form.get('bmi')
        heart_rate = request.form.get('heart_rate')
        smoking_status = request.form.get('smoking_status')

        # Convert form data to proper types
        age = float(age)
        hypertension = int(hypertension)
        heart_disease = int(heart_disease)
        ever_married = 1 if ever_married == 'Yes' else 0
        avg_glucose_level = float(avg_glucose_level)
        bmi = float(bmi)
        heart_rate = float(heart_rate)

        # Prepare the data for the RandomForest model in a DataFrame
        medical_features = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'ever_married': [ever_married],
            'work_type': [work_type],
            'residence_type': [residence_type],
            'avg_glucose_level': [avg_glucose_level],
            'bmi': [bmi],
            'heart_rate': [heart_rate],
            'smoking_status': [smoking_status]
        })

        # Log the DataFrame for debugging
        print(f"Prepared Medical Features DataFrame:\n{medical_features}")

        # Convert the DataFrame to a NumPy array before passing it to the model
        medical_history_pred = gb_model.predict_proba(medical_features.values)  # Use .values to get the NumPy array
        stroke_probability_medical = medical_history_pred[0][1]  # Probability of stroke

        print(f"Stroke probability from medical history model: {stroke_probability_medical}")

    except ValueError as e:
        return jsonify({'error': f'Invalid medical history data: {str(e)}'}), 400

    # Run inference with the image model
    try:
        image = Image.open(filepath).convert('RGB')
        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = facial_model(image)
            facial_probabilities = torch.nn.functional.softmax(output, dim=1)
            stroke_probability_facial = facial_probabilities[0][1].item()  # Probability of stroke
        print(f"Stroke probability from facial model: {stroke_probability_facial}")
    except Exception as e:
        return jsonify({'error': f'Model inference error (facial): {str(e)}'}), 500

    # Combine the probabilities (weighted sum)
    combined_stroke_probability = (stroke_probability_facial * 0.6) + (stroke_probability_medical * 0.4)
    combined_no_stroke_probability = 1 - combined_stroke_probability

    # Return the combined probabilities as JSON
    return jsonify({
        'stroke_probability': round(combined_stroke_probability * 100, 2),
        'no_stroke_probability': round(combined_no_stroke_probability * 100, 2)
    })

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    app.run(port=3003, debug=True)
