from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import os
import uuid
import io
import torch
import argparse
from stable_diffusion_tf.stable_diffusion import StableDiffusion
import uvicorn

app = FastAPI()

TEMP_FOLDER = "temp"
DOWNLOAD_FOLDER = "download"

def create_folders():
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

@app.post("/generate_image")
async def generate_image(prompt: str = "cartoon portrait", image: UploadFile = File(...), num_steps: int = 100):
    # Save the uploaded image
    img = Image.open(io.BytesIO(await image.read()))
    img.save("input_image.jpeg")

    # Generate the modified image
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, nargs="?", default=prompt, help="the prompt to render")
    parser.add_argument("--input", type=str, nargs="?", required=True, help="the input image filename")
    args = parser.parse_args([f"--prompt={prompt}", "--input=input_image.jpeg"])

    generator = StableDiffusion(
        img_height=512,
        img_width=512,
        jit_compile=False
    )

    modified_img = generator.generate(
        args.prompt,
        num_steps=num_steps,  # Adjust this parameter for image quality (higher for higher quality, but more time-consuming)
        unconditional_guidance_scale=7.5,  # You can adjust this parameter for stylization if needed
        temperature=1,
        batch_size=1,
        input_image=args.input,
        input_image_strength=0.8
    )

    # Save the modified image
    output_filename = f"generated_{uuid.uuid4().hex}.png"
    output_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    Image.fromarray(modified_img[0]).save(output_path)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}

@app.get("/download/{file_path}")
def download_file(file_path: str):
    image_path = os.path.join(DOWNLOAD_FOLDER, file_path)
    return FileResponse(image_path, media_type="image/png")

@app.on_event("startup")
async def startup_event():
    create_folders()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
