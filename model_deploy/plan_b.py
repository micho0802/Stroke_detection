#Implementation based on this: https://huggingface.co/ContactDoctor/Bio-Medical-MultiModal-Llama-3-8B-V1

from huggingface_hub import login

login(token="API_Key", add_to_git_credential = True)

import torch
from PIL import Image
import gradio as gr
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from torchvision import models, transforms



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

# Placeholder function for the stroke prediction model
def stroke_prediction_model(image):
    # Define the ResNet50 model with a modified fully connected layer (2 output classes)
    facial_model = models.resnet50(weights=None)
    facial_model.fc = torch.nn.Linear(in_features=2048, out_features=2)
    facial_model.load_state_dict(torch.load('/home/mich02/Desktop/UMKC_DS_Capstone/model_pretrained/facial_stroke_model.pth', weights_only=True)) #Pretrain model can be found in https://huggingface.co/micho02/Face_stroke_detection/tree/main
    facial_model.eval()
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Apply transformations to the input image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Get stroke probability
    with torch.no_grad():
        outputs = facial_model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        stroke_probability = probabilities[0, 1].item()  # Probability of the stroke class

    # Format the result as a percentage
    return f"Stroke Probability: {stroke_probability:.2%}"

# Function for multimodal inference
def multimodal_inference(image, question, stroke_prediction=False):
    # Check if stroke prediction is requested
    if stroke_prediction:
        # Run the stroke prediction model if the checkbox is checked
        return stroke_prediction_model(image)
    
    # Use a blank image if none is provided
    if image is None:
        image = Image.new("RGB", (224, 224), color="white")
    
    # Ensure image is in RGB format
    image = image.convert("RGB") if image.mode != "RGB" else image
    
    # Prepare input message
    msgs = [{'role': 'user', 'content': [image, question]}]

    # Generate the response using the multimodal Llama model
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

# Define the Gradio interface
iface = gr.Interface(
    fn=multimodal_inference,
    inputs=[
        gr.Image(type="pil", label="Upload Facial Image"),
        gr.Textbox(label="Enter your question"),
        gr.Checkbox(label="What’s the probability of having a stroke?")  # Checkbox for stroke prediction
    ],
    outputs="text",
    title="Bio-Medical Multimodal Llama Model with Stroke Prediction",
    description="Upload an image and enter a question. Select the checkbox for stroke probability prediction, or leave it unchecked for general insights."
)

# Launch the Gradio app
iface.launch()