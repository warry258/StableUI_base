import os
import subprocess
import torch
import numpy as np
import random
import gradio as gr
from diffusers import StableDiffusionXLPipeline
from PIL import Image

# Constants
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344
MODEL_PATH = '/content/StableUI_base/model_link.safetensors'
LORA_PATH = '/content/StableUI_base/lora_model.safetensors'

# Download model
def download_model():
    os.makedirs("/content/StableUI_base", exist_ok=True)
    subprocess.run([
        "wget", "-O", MODEL_PATH,
        "https://civitai.com/api/download/models/128078?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    ])

# Clear console
def clear_console():
    os.system('clear')

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
download_model()
pipe = StableDiffusionXLPipeline.from_single_file(MODEL_PATH, use_safetensors=True, torch_dtype=torch.float16).to(device)
clear_console()
print("\033[1;32mModel loaded!\033[0m")

# Create a placeholder image
def create_placeholder_image():
    return Image.new('RGB', (512, 512), color=(73, 109, 137))

# Load and apply LoRA
def apply_lora(pipe, use_lora, lora_scale):
    if use_lora:
        pipe.load_lora_weights('/content/StableUI_base', weight_name='lora_model.safetensors')
        pipe.fuse_lora(lora_scale)
        return f"LoRA applied with scale: {lora_scale}"
    else:
        pipe.unfuse_lora()
        return "LoRA not applied"

# Inference function
def infer(prompt, seed, width, height, guidance_scale, num_inference_steps, use_lora, lora_scale, progress=gr.Progress(track_tqdm=True)):
    if seed == -1:  # -1 indicates random seed
        seed = random.randint(0, MAX_SEED)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    lora_message = apply_lora(pipe, use_lora, lora_scale)
    
    progress(0, desc="Generating image")
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        width=width, 
        height=height,
        generator=generator,
    ).images[0]
    
    progress(1, desc="Done")
    return image, seed, lora_message

# UI setup
css = """
#col-container {
    margin: 0 auto;
    max-width: 580px;
}
footer {
    display: none !important;
}
"""

examples = [
    "a cat",
    "a cat in the hat",
    "a cat in the cowboy hat",
]

with gr.Blocks(css=css, theme='ParityError/Interstellar') as app:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""
    # Stable Diffusion <a href="https://www.patreon.com/marat_ai">by marat_ai</a> 
    <a href="https://www.youtube.com/@marat_ai">
        <img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube" style="display: inline;"/>
    </a>
    <a href="https://www.patreon.com/marat_ai">
        <img src="https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white" alt="Patreon" style="display: inline;"/>
    </a>

    Google Colab's free tier offers about 4 hours of GPU usage per day. No authorization, no data storing or tracking. Your session data will be deleted when this session closes.
""")

        with gr.Group():
            with gr.Row():
                prompt = gr.Text(label="Prompt", show_label=False, lines=1, max_lines=7,
                                 placeholder="Enter your prompt", container=False, scale=4)
                run_button = gr.Button("🚀 Run", scale=1, variant='primary')
        
        result = gr.Image(label="Result", show_label=False, value=create_placeholder_image())
        
        with gr.Group():
            with gr.Accordion("⚙️ Settings", open=False):
                seed = gr.Slider(label="Seed (-1 for random)", minimum=-1, maximum=MAX_SEED, step=1, value=-1)
                
                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                    height = gr.Slider(label="Height", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=5.0)
                    num_inference_steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=20)

        with gr.Accordion("🧬 LoRA Settings", open=False):
            use_lora = gr.Checkbox(label="Use LoRA", value=False)
            lora_scale = gr.Slider(label="LoRA Scale", minimum=0.0, maximum=1.0, step=0.01, value=0.5)

        seed_used = gr.Number(label="Seed used", interactive=False)
        lora_message = gr.Text(label="LoRA Status", interactive=False)

        gr.Examples(examples=examples, inputs=[prompt])
    
    run_button.click(
        fn=infer,
        inputs=[prompt, seed, width, height, guidance_scale, num_inference_steps, use_lora, lora_scale],
        outputs=[result, seed_used, lora_message],
        show_progress=True
    )

if __name__ == "__main__":
    app.launch(share=True, inline=False, inbrowser=False, debug=True)
