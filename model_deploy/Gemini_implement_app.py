import os
import time
from typing import List, Tuple, Optional

import google.generativeai as genai
import gradio as gr
from PIL import Image

print("google-generativeai:", genai.__version__)

GOOGLE_API_KEY = "Secret API"

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

def preprocess_stop_sequences(stop_sequences: str) -> Optional[List[str]]:
    return [seq.strip() for seq in stop_sequences.split(",")] if stop_sequences else None

def preprocess_image(image: Image.Image) -> Image.Image:
    image_height = int(image.height * IMAGE_WIDTH / image.width)
    return image.resize((IMAGE_WIDTH, image_height))

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
    inputs = [text_prompt] if image_prompt is None else [text_prompt, preprocess_image(image_prompt)]
    
    response = model.generate_content(inputs, stream=True, generation_config=generation_config)
    response.resolve()

    chatbot[-1][1] = ""
    for chunk in response:
        for i in range(0, len(chunk.text), 10):
            chatbot[-1][1] += chunk.text[i:i + 10]
            time.sleep(0.01)
            yield chatbot

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