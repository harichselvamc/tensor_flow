from fastapi import FastAPI, UploadFile, File
from PIL import Image
import os
import rembg
import cv2
import numpy as np
from io import BytesIO
import torch
from stable_diffusion_tf.stable_diffusion import StableDiffusion
import io
import uuid

app = FastAPI()

TEMP_FOLDER = "temp"
OUTPUT_FOLDER = "output"
DOWNLOAD_FOLDER = "download"

def create_folders():
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

@app.post("/solution")
async def solution_endpoint(prompt: str = "cartoon portrait", file: UploadFile = File(...)):
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Perform background removal using rembg library
    with open(file_path, "rb") as f:
        image_data = f.read()
        output_data = rembg.remove(image_data)

    output_filename = file.filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    # Save the output image
    with open(output_path, "wb") as f:
        f.write(output_data)

        # Resize the image to 1080x1080
        image = Image.open(output_path)
        resized_image = image.resize((1080, 1080))
        resized_image.save(output_path)

    # Generate the modified image
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, nargs="?", default=prompt, help="the prompt to render")
    parser.add_argument("--input", type=str, nargs="?", required=True, help="the input image filename")
    args = parser.parse_args([f"--prompt={prompt}", f"--input={output_path}"])

    generator = StableDiffusion(
        img_height=512,
        img_width=512,
        jit_compile=False
    )

    modified_img = generator.generate(
        args.prompt,
        num_steps=10,
        unconditional_guidance_scale=7.5,
        temperature=1,
        batch_size=1,
        input_image=args.input,
        input_image_strength=0.8
    )

    # Save the modified image
    output_image_path = "output_image.jpeg"
    Image.fromarray(modified_img[0]).save(output_image_path)

    # Save the modified image to the download folder
    output_filename = f"generated_{uuid.uuid4().hex}.png"
    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    Image.fromarray(modified_img[0]).save(download_path)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}

@app.get("/download/{file_path}")
def download_file(file_path: str):
    image_path = os.path.join(DOWNLOAD_FOLDER, file_path)
    return FileResponse(image_path, media_type="image/png")

@app.on_event("startup")
async def startup_event():
    create_folders()

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up temporary and output directories
    for file_name in os.listdir(TEMP_FOLDER):
        file_path = os.path.join(TEMP_FOLDER, file_name)
        os.remove(file_path)

    for file_name in os.listdir(OUTPUT_FOLDER):
        file_path = os.path.join(OUTPUT_FOLDER, file_name)
        os.remove(file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
