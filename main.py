import os
import subprocess
import torch
import numpy as np
import gradio as gr
from diffusers import StableDiffusionXLPipeline

# Константы
MAX_SEED = np.iinfo(np.int32).max
MAX_IMAGE_SIZE = 1344
MODEL_PATH = '/content/StableUI/model_link.safetensors'

# Загрузка модели
def download_model():
    os.makedirs("/content/StableUI", exist_ok=True)
    subprocess.run([
        "wget", "-O", MODEL_PATH,
        "https://civitai.com/api/download/models/128078?type=Model&format=SafeTensor&size=pruned&fp=fp16"
    ])

# Очистка консоли
def clear_console():
    os.system('clear')

# Настройка
device = "cuda" if torch.cuda.is_available() else "cpu"

# Загрузка модели
download_model()
pipe = StableDiffusionXLPipeline.from_single_file(MODEL_PATH, use_safetensors=True, torch_dtype=torch.float16).to(device)
clear_console()
print("\033[1;32mМодель загружена!\033[0m")

# Функция вывода
def infer(prompt, seed, width, height, guidance_scale, num_inference_steps, progress=gr.Progress(track_tqdm=True)):
    generator = torch.Generator(device=device).manual_seed(seed)
    
    progress(0, desc="Генерация изображения")
    image = pipe(
        prompt=prompt,
        guidance_scale=guidance_scale, 
        num_inference_steps=num_inference_steps, 
        width=width, 
        height=height,
        generator=generator,
    ).images[0]
    
    progress(1, desc="Готово")
    return image

# Настройка UI
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
        gr.Markdown("""
    # Stable Diffusion <a href="https://www.patreon.com/marat_ai">by marat_ai</a> 
    <a href="https://www.youtube.com/@marat_ai">
        <img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube" style="display: inline;"/>
    </a>
    <a href="https://www.patreon.com/marat_ai">
        <img src="https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white" alt="Patreon" style="display: inline;"/>
    </a>
""")

        with gr.Group():
            with gr.Row():
                prompt = gr.Text(label="Prompt", show_label=False, lines=1, max_lines=7,
                                 placeholder="Введите ваш запрос", container=False, scale=4)
                run_button = gr.Button("🚀 Запустить", scale=1, variant='primary')
        
        result = gr.Image(label="Результат", show_label=False)
        
        with gr.Group():
            with gr.Accordion("⚙️ Настройки", open=False):
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                
                with gr.Row():
                    width = gr.Slider(label="Ширина", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                    height = gr.Slider(label="Высота", minimum=256, maximum=MAX_IMAGE_SIZE, step=64, value=1024)
                
                with gr.Row():
                    guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=5.0)
                    num_inference_steps = gr.Slider(label="Шаги", minimum=1, maximum=50, step=1, value=20)

        gr.Examples(examples=examples, inputs=[prompt])
    
    run_button.click(
        fn=infer,
        inputs=[prompt, seed, width, height, guidance_scale, num_inference_steps],
        outputs=result,
        show_progress=True
    )

if __name__ == "__main__":
    app.launch(share=True, inline=False, inbrowser=False, debug=True)
