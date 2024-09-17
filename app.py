import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import gradio as gr

# Load the model
model_id = "CompVis/stable-diffusion-v1-4"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    return pipe.to(device)

# Load the model once
pipe = load_model()

# Function to generate thumbnail
def generate_thumbnail(prompt, width, height):
    # Generate image using Stable Diffusion
    image = pipe(prompt).images[0]
    
    # Resize the image to user-specified dimensions
    thumbnail_size = (int(width), int(height))
    image = image.resize(thumbnail_size)

    return image  # Return as PIL image for Gradio

# Gradio interface
inputs = [
    gr.Textbox(label="Enter your prompt for the YouTube thumbnail", placeholder="A beautiful landscape with mountains and sunset"),
    gr.Slider(label="Width (px)", minimum=640, maximum=1920, step=1, value=1280),
    gr.Slider(label="Height (px)", minimum=360, maximum=1080, step=1, value=720)
]
outputs = gr.Image(type="pil", label="Generated YouTube Thumbnail")  # Change type to "pil"

# Create Gradio Interface
demo = gr.Interface(
    fn=generate_thumbnail, 
    inputs=inputs, 
    outputs=outputs, 
    title="YouTube Thumbnail Generator",
    description="Generate YouTube thumbnails based on text prompts using Stable Diffusion.",
)

# Launch the app
demo.launch(share=True)
