import os
import time
from typing import List, Tuple, Optional

import google.generativeai as genai
import gradio as gr
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn.functional as F

import joblib
import networkx as nx
import pandas as pd

# Create a new directed graph for stroke risks, symptoms, prevention, and treatment
G = nx.DiGraph()

# Add nodes for risk factors, prevention, and treatment
G.add_node("Stroke", type="condition")
G.add_node("High Blood Pressure", type="risk_factor")
G.add_node("Smoking", type="risk_factor")
G.add_node("Diabetes", type="risk_factor")
G.add_node("Obesity", type="risk_factor")

G.add_node("Aspirin Therapy", type="treatment")
G.add_node("Clot-dissolving Medication", type="treatment")
G.add_node("Physical Therapy", type="treatment")
G.add_node("Lifestyle Changes", type="prevention")
G.add_node("Smoking Cessation", type="prevention")
G.add_node("Healthy Diet", type="prevention")
G.add_node("Exercise", type="prevention")

# Add edges representing relationships between risks, symptoms, prevention, and treatments
G.add_edge("Stroke", "High Blood Pressure", relationship="risk")
G.add_edge("Stroke", "Smoking", relationship="risk")
G.add_edge("Stroke", "Diabetes", relationship="risk")
G.add_edge("Stroke", "Obesity", relationship="risk")

G.add_edge("High Blood Pressure", "Lifestyle Changes", relationship="prevention")
G.add_edge("Smoking", "Smoking Cessation", relationship="prevention")
G.add_edge("Diabetes", "Healthy Diet", relationship="prevention")
G.add_edge("Obesity", "Exercise", relationship="prevention")

G.add_edge("Stroke", "Aspirin Therapy", relationship="treatment")
G.add_edge("Stroke", "Clot-dissolving Medication", relationship="treatment")
G.add_edge("Stroke", "Physical Therapy", relationship="treatment")

# Function to get recommendations from the knowledge graph
def get_stroke_recommendations():
    recommendations = [n for n in G.neighbors("Stroke") if G.nodes[n]['type'] in ['prevention', 'treatment']]
    return recommendations



print("google-generativeai:", genai.__version__)

GOOGLE_API_KEY = "Secret_API" #https://ai.google.dev/gemini-api

TITLE = """<h1 align="center">🕹️ Google Gemini Chatbot 🔥</h1>"""
SUBTITLE = """<h2 align="center">🎨Create with Multimodal Gemini</h2>"""
DUPLICATE = """
<div style="text-align: center; display: flex; justify-content: center; align-items: center;">
    <a href="https://huggingface.co/spaces/Rahatara/build_with_gemini/blob/main/allgemapp.py?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space" style="margin-right: 10px;">
    </a>
    <span>Duplicate the Space and run securely with your 
        <a href="https://makersuite.google.com/app/apikey">GOOGLE API KEY</a>.
    </span>
</div>
"""
IMAGE_WIDTH = 512

# Load ResNet50 model for facial stroke detection
facial_model = models.resnet50(weights=None)
facial_model.fc = torch.nn.Linear(in_features=2048, out_features=2)
facial_model.load_state_dict(torch.load('/home/mich02/Desktop/UMKC_DS_Capstone/model_pretrained/facial_stroke_model.pth', weights_only=True))
facial_model.eval()


# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def preprocess_stop_sequences(stop_sequences: str) -> Optional[List[str]]:
    return [seq.strip() for seq in stop_sequences.split(",")] if stop_sequences else None

def preprocess_image(image: Image.Image) -> Image.Image:
    image_height = int(image.height * IMAGE_WIDTH / image.width)
    return image.resize((IMAGE_WIDTH, image_height))

def stroke_detection(image: Image.Image) -> Tuple[str, float, List[str]]:
    """Perform stroke detection using the ResNet50 model and return the result and recommendations."""
    image = transform(image).unsqueeze(0)  # Prepare image for the model (add batch dimension)
    with torch.no_grad():
        output = facial_model(image)
        probabilities = F.softmax(output, dim=1)
        stroke_probability = probabilities[0][1].item()
    
    # Get stroke prevention and treatment recommendations if stroke probability is high
    recommendations = []
    if stroke_probability >= 0.90:
        recommendations = get_stroke_recommendations()
    
    return f"Stroke Probability: {stroke_probability * 100:.2f}%", stroke_probability, recommendations

# Load the Random Forest for medical history
medical_history_model = joblib.load('/home/mich02/Desktop/UMKC_DS_Capstone/model_pretrained/medical_history_model.pth')

# Preprocess input for medical history model (make sure this matches RandomForest model)
def preprocess_medical_data(medical_data: dict) -> List[float]:
    """Extract and return the complete feature set for the RandomForestClassifier."""
    
    # Ensure all features are present, and if missing, provide default values
    features = [
        medical_data.get("id", 0),
        medical_data.get('age', 0),  # Use 0 or another default value if missing
        medical_data.get('hypertension', 0),
        medical_data.get('heart_disease', 0),
        medical_data.get('avg_glucose_level', 0.0),
        medical_data.get('bmi', 0.0),
        medical_data.get('smoking_status', 0),
        medical_data.get("gender", 0),
        medical_data.get('work_type', 0),  # Example of additional features
        medical_data.get('Residence_type', 0),
        medical_data.get('ever_married', 0)

    ]
    
    # Ensure the length matches the expected 11 features
    if len(features) != 11:
        raise ValueError(f"Expected 11 features, but got {len(features)}")
    
    return features

def multimodal_stroke_detection(image: Image.Image, medical_data: dict) -> Tuple[str, float, List[str]]:
    """Perform multimodal stroke detection using facial and medical history models."""
    
    # 1. Facial stroke detection
    image = transform(image).unsqueeze(0)  # Prepare image for the model (add batch dimension)
    with torch.no_grad():
        output = facial_model(image)
        facial_probabilities = F.softmax(output, dim=1)
        facial_stroke_probability = facial_probabilities[0][1].item()

    # 2. Medical history model prediction
    medical_features = preprocess_medical_data(medical_data)  # Convert to features list
    medical_stroke_probability = medical_history_model.predict_proba([medical_features])[0][1]  # Binary classification

    # 3. Combine the two model predictions
    combined_probability = 0.6 * facial_stroke_probability + 0.4 * medical_stroke_probability

    # 4. Get recommendations if the combined stroke probability is high
    recommendations = []
    if combined_probability >= 0.90:
        recommendations = get_stroke_recommendations()

    return f"Combined Stroke Probability: {combined_probability * 100:.2f}%", combined_probability, recommendations

def user(text_prompt: str, chatbot: List[Tuple[str, str]]):
    return "", chatbot + [[text_prompt, None]]

def bot(
    google_key: str,
    image_prompt: Optional[Image.Image],
    medical_data_prompt: Optional[dict],
    temperature: float,
    max_output_tokens: int,
    stop_sequences: str,
    top_k: int,
    top_p: float,
    chatbot: Optional[List[dict]]  # Chatbot messages should be formatted as dictionaries with "role" and "content"
):
    # Initialize the chatbot if it's None
    if chatbot is None:
        chatbot = []

    # Check if the chatbot is empty and add a system greeting
    if not chatbot:
        chatbot.append({"role": "system", "content": "Hello! How can I assist you today?"})

    # Validate the Google API key and configure Generative AI API
    if google_key:
        genai.configure(api_key=GOOGLE_API_KEY)
    else:
        chatbot.append({"role": "assistant", "content": "No API key provided. Please provide a valid Google API key."})
        yield chatbot
        return

    # Handle image input and medical data prediction
    if image_prompt and medical_data_prompt:
        try:
            result, combined_probability, recommendations = multimodal_stroke_detection(image_prompt, medical_data_prompt)
            chatbot.append({"role": "assistant", "content": f"Prediction result: {result}"})
            if combined_probability >= 0.90:
                chatbot.append({"role": "assistant", "content": "The stroke probability is high. Seek medical attention."})
                chatbot.append({"role": "assistant", "content": f"Recommended actions: {', '.join(recommendations)}"})
            yield chatbot
        except Exception as e:
            chatbot.append({"role": "assistant", "content": f"Error: {str(e)}"})
            yield chatbot
        return

    # Handle text-based conversation with Gemini
    text_prompt = chatbot[-1]["content"]  # Get the user's last input
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        stop_sequences=preprocess_stop_sequences(stop_sequences),
        top_k=top_k,
        top_p=top_p,
    )

    model_name = "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name)
    inputs = [text_prompt]

    response = model.generate_content(inputs, stream=True, generation_config=generation_config)
    response.resolve()

    # Properly format the chatbot response
    chatbot.append({"role": "assistant", "content": ""})
    for chunk in response:
        chatbot[-1]["content"] += chunk.text  # Append chunked responses
        time.sleep(0.01)
        yield chatbot  # Update the chatbot with the assistant's response in real-time



# Gradio UI Components
google_key_component = gr.Textbox(
    label="GOOGLE API KEY",
    type="password",
    placeholder="...",
    visible=GOOGLE_API_KEY is None
)

image_prompt_component = gr.Image(type="pil", label="Image")
chatbot_component = gr.Chatbot(label='Gemini', type='messages', bubble_full_width=False)
text_prompt_component = gr.Textbox(placeholder="Hi there!", label="Ask me anything and press Enter")
run_button_component = gr.Button("Run")
temperature_component = gr.Slider(minimum=0, maximum=1.0, value=0.4, step=0.05, label="Temperature")
max_output_tokens_component = gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Token limit")
stop_sequences_component = gr.Textbox(label="Add stop sequence", placeholder="STOP, END")
top_k_component = gr.Slider(minimum=1, maximum=40, value=32, step=1, label="Top-K")
top_p_component = gr.Slider(minimum=0, maximum=1, value=1, step=0.01, label="Top-P")
medical_data_prompt_component = gr.JSON(
    label="Medical History (Provide details in JSON format)", 
    value={
        "age": 67,
        "hypertension": 0,
        "heart_disease": 1,
        "avg_glucose_level": 228.69,
        "bmi": 36.6,
        "smoking_status": 1,  # Example values
        "gender": 1,
        'work_type': 0,
        'Residence_type': 0,
        'ever_married': 0,
        "id": 0
    }
)

user_inputs = [text_prompt_component, chatbot_component]
bot_inputs = [google_key_component, image_prompt_component, medical_data_prompt_component, temperature_component, max_output_tokens_component, stop_sequences_component, top_k_component, top_p_component, chatbot_component]

with gr.Blocks() as demo:
    gr.HTML(TITLE)
    gr.HTML(SUBTITLE)
    gr.HTML(DUPLICATE)
    with gr.Column():
        google_key_component.render()
        with gr.Row():
            image_prompt_component.render()  # Image input for facial recognition
            chatbot_component.render()  # Chatbot for interaction
        medical_data_prompt_component.render()  # Medical history input
        text_prompt_component.render()  # Text-based input for chatbot
        run_button_component.render()
        with gr.Accordion("Parameters", open=False):
            temperature_component.render()
            max_output_tokens_component.render()
            stop_sequences_component.render()
            with gr.Accordion("Advanced", open=False):
                top_k_component.render()
                top_p_component.render()

    run_button_component.click(fn=user, inputs=user_inputs, outputs=[text_prompt_component, chatbot_component], queue=False).then(fn=bot, inputs=bot_inputs, outputs=[chatbot_component])
    text_prompt_component.submit(fn=user, inputs=user_inputs, outputs=[text_prompt_component, chatbot_component], queue=False).then(fn=bot, inputs=bot_inputs, outputs=[chatbot_component])

demo.launch(share=True)
