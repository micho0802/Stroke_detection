import os
import time
from typing import List, Tuple, Optional

import google.generativeai as genai
import gradio as gr
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn.functional as F

print("google-generativeai:", genai.__version__)

GOOGLE_API_KEY = "Secret_API"

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

# Load your ResNet50 model for stroke detection
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=2)
model.load_state_dict(torch.load('facial_stroke_model.pth'))
model.eval()

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

def stroke_detection(image: Image.Image) -> Tuple[str, float]:
    """Perform stroke detection using the ResNet50 model and return the result."""
    image = transform(image).unsqueeze(0)  # Prepare image for the model (add batch dimension)
    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output, dim=1)
        stroke_probability = probabilities[0][1].item()
    
    return f"Stroke Probability: {stroke_probability * 100:.2f}%", stroke_probability

def user(text_prompt: str, chatbot: List[Tuple[str, str]]):
    return "", chatbot + [[text_prompt, None]]

def bot(
    google_key: str,
    image_prompt: Optional[Image.Image],
    temperature: float,
    max_output_tokens: int,
    stop_sequences: str,
    top_k: int,
    top_p: float,
    chatbot: List[Tuple[str, str]]
):
    google_key = google_key or GOOGLE_API_KEY
    if not google_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please set it up.")

    # Check if an image is provided for stroke detection
    if image_prompt:
        result, stroke_probability = stroke_detection(image_prompt)
        chatbot[-1][1] = result
        if stroke_probability >= 0.90:
            chatbot.append(["", "The stroke probability is very high. Please seek immediate medical attention or call emergency services."])
        yield chatbot
        return

    # If no image is provided, handle text-based conversation with Gemini
    text_prompt = chatbot[-1][0]
    genai.configure(api_key=google_key)
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

    chatbot[-1][1] = ""
    for chunk in response:
        for i in range(0, len(chunk.text), 10):
            chatbot[-1][1] += chunk.text[i:i + 10]
            time.sleep(0.01)
            yield chatbot

# Gradio UI Components
google_key_component = gr.Textbox(
    label="GOOGLE API KEY",
    type="password",
    placeholder="...",
    visible=GOOGLE_API_KEY is None
)

image_prompt_component = gr.Image(type="pil", label="Image")
chatbot_component = gr.Chatbot(label='Gemini', bubble_full_width=False)
text_prompt_component = gr.Textbox(placeholder="Hi there!", label="Ask me anything and press Enter")
run_button_component = gr.Button("Run")
temperature_component = gr.Slider(minimum=0, maximum=1.0, value=0.4, step=0.05, label="Temperature")
max_output_tokens_component = gr.Slider(minimum=1, maximum=2048, value=1024, step=1, label="Token limit")
stop_sequences_component = gr.Textbox(label="Add stop sequence", placeholder="STOP, END")
top_k_component = gr.Slider(minimum=1, maximum=40, value=32, step=1, label="Top-K")
top_p_component = gr.Slider(minimum=0, maximum=1, value=1, step=0.01, label="Top-P")

user_inputs = [text_prompt_component, chatbot_component]
bot_inputs = [google_key_component, image_prompt_component, temperature_component, max_output_tokens_component, stop_sequences_component, top_k_component, top_p_component, chatbot_component]

with gr.Blocks() as demo:
    gr.HTML(TITLE)
    gr.HTML(SUBTITLE)
    gr.HTML(DUPLICATE)
    with gr.Column():
        google_key_component.render()
        with gr.Row():
            image_prompt_component.render()
            chatbot_component.render()
        text_prompt_component.render()
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

demo.launch()
