#Implementation based on this: https://huggingface.co/ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1

from huggingface_hub import login

login(token="API_Key", add_to_git_credential = True)

import torch
from PIL import Image
import gradio as gr
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# Load the model and tokenizer
model = AutoModel.from_pretrained(
    "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"  # Use "flash_attention_2" if FlashAttention is installed
)

tokenizer = AutoTokenizer.from_pretrained(
    "ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1",
    trust_remote_code=True
)

# Function for Gradio to process inputs and generate responses
from PIL import Image

def multimodal_inference(image, question):
    # Use a blank image if no image is provided
    if image is None:
        # Create a blank white image (e.g., 224x224 pixels)
        image = Image.new("RGB", (224, 224), color="white")
    
    # Ensure image is in RGB format
    image = image.convert("RGB") if image.mode != "RGB" else image
    
    # Prepare input message
    msgs = [{'role': 'user', 'content': [image, question]}]

    # Generate the response
    try:
        res = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.95,
            stream=False
        )
        
        # Concatenate the generated text
        generated_text = "".join(res)
        return generated_text
    except Exception as e:
        return f"Error: {str(e)}"


    
# Gradio interface
iface = gr.Interface(
    fn=multimodal_inference,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Enter your question")],
    outputs="text",
    title="Bio-Medical Multimodal Llama Model",
    description="Upload an image and enter a medical-related question to get insights."
)

# Launch the Gradio app
iface.launch()
