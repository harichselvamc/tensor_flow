from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import tempfile
from PIL import Image
from io import BytesIO
import requests
from diffusers import StableDiffusionImg2ImgPipeline
import torch

# Initialize the pipeline
device = "cpu"
model_id_or_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id_or_path, torch_dtype=torch.float32)
pipe = pipe.to(device)

app = FastAPI()


@app.post("/generate/")
async def generate_image(prompt: str, image_url: str):
    # Download the image from the URL
    response = requests.get(image_url)
    init_image = Image.open(BytesIO(response.content)).convert("RGB")
    init_image = init_image.resize((768, 512))

    # Generate the image using the pipeline
    images = pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5).images

    # Save the generated image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image_path = temp_file.name
        images[0].save(image_path)

    # Return the generated image as a file response
    return FileResponse(image_path, media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
