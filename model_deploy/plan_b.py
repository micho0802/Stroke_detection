#Implementation based on this: https://huggingface.co/ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1

from huggingface_hub import login

login(token="Secret_API", add_to_git_credential = True)

# Import necessary libraries
import pandas as pd
import numpy as np
import torch
from PIL import Image
import gradio as gr
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from torchvision import models, transforms
import joblib

# Load the ensemble model
xgb_model = joblib.load("/home/mich02/Desktop/UMKC_DS_Capstone/model_pretrained/XGBoost_best_model.pkl")

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the multimodal model and tokenizer
model = AutoModel.from_pretrained(
    "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained(
    "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1",
    trust_remote_code=True
)

# Placeholder function for the facial stroke prediction model
def facial_stroke_prediction_model(image):
    facial_model = models.resnet50(weights=None)
    facial_model.fc = torch.nn.Linear(in_features=2048, out_features=2)
    facial_model.load_state_dict(torch.load('/home/mich02/Desktop/UMKC_DS_Capstone/model_pretrained/facial_stroke_model.pth', weights_only=True))
    facial_model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = facial_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        stroke_probability = probabilities[0, 1].item()
    return stroke_probability

# Medical history stroke prediction model function
def medical_history_prediction_model(gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })
    
    # Ensure the features align with those used in training
    input_data_encoded = pd.get_dummies(input_data, drop_first=True)
    input_data_encoded = input_data_encoded.reindex(columns=xgb_model.feature_names_in_, fill_value=0)
    
    # Predict stroke probability
    probability = xgb_model.predict_proba(input_data_encoded)[0][1]
    return probability

# Combine both predictions
def combined_stroke_probability(image, gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status):
    # Get stroke probabilities from facial and medical history models
    facial_probability = facial_stroke_prediction_model(image)
    medical_probability = medical_history_prediction_model(gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status)
    
    # Combine using weighted average
    combined_probability = facial_probability * 0.6 + medical_probability * 0.4
    return f"Combined Stroke Probability: {combined_probability:.2%}"

# Function for multimodal inference
def multimodal_inference(image, question, stroke_prediction, gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status):
    if stroke_prediction:
        return combined_stroke_probability(image, gender, age, hypertension, heart_disease, avg_glucose_level, bmi, smoking_status)
    
    if image is None:
        image = Image.new("RGB", (224, 224), color="white")
    
    image = image.convert("RGB") if image.mode != "RGB" else image
    msgs = [{'role': 'user', 'content': [image, question]}]

    try:
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.95,
            stream=False
        )
        generated_text = "".join(res)
        return generated_text
    except Exception as e:
        return f"Error: {str(e)}"

# Define Gradio interface with additional medical history inputs
iface = gr.Interface(
    fn=multimodal_inference,
    inputs=[
        gr.Image(type="pil", label="Upload Facial Image"),
        gr.Textbox(label="Enter your question"),
        gr.Checkbox(label="What’s the probability of having a stroke?"),
        gr.Dropdown(choices=["Male", "Female"], label="Gender"),
        gr.Slider(0, 100, label="Age"),
        gr.Checkbox(label="Hypertension"),
        gr.Checkbox(label="Heart Disease"),
        gr.Number(label="Average Glucose Level"),
        gr.Number(label="BMI"),
        gr.Dropdown(choices=["never smoked", "formerly smoked", "smokes"], label="Smoking Status")
    ],
    outputs="text",
    title="Bio-Medical Multimodal Llama Model with Stroke Prediction",
    description="Upload an image and enter a question. Select the checkbox for stroke probability prediction, or leave it unchecked for general insights."
)

# Launch the Gradio app
iface.launch()
