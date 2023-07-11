# import os
# import uvicorn
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import FileResponse
# from PIL import Image
# import io
# import argparse
# from stable_diffusion_tf.stable_diffusion import StableDiffusion

# app = FastAPI()

# @app.post("/generate_image")
# async def generate_image(prompt: str, image: UploadFile = File(...)):
#     # Save the uploaded image
#     img = Image.open(io.BytesIO(await image.read()))
#     input_image_path = "input_image.jpeg"
#     img.save(input_image_path)

#     # Generate the modified image
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--prompt", type=str, nargs="?", required=True, help="the prompt to render")
#     parser.add_argument("--input", type=str, nargs="?", required=True, help="the input image filename")
#     args = parser.parse_args([f"--prompt={prompt}", f"--input={input_image_path}"])

#     generator = StableDiffusion(
#         img_height=512,
#         img_width=512,
#         jit_compile=False
#     )

#     modified_img = generator.generate(
#         args.prompt,
#         num_steps=50,
#         unconditional_guidance_scale=7.5,
#         temperature=1,
#         batch_size=1,
#         input_image=args.input,
#         input_image_strength=0.8
#     )

#     # Save the modified image
#     output_image_path = "download/output_image.jpeg"
#     Image.fromarray(modified_img[0]).save(output_image_path)

#     # Create a download link for the image
#     output_filename = "output_image.jpeg"
#     image_link = f"http://localhost:8000/download/{output_filename}"

#     return {"link": image_link}

# @app.get("/download/{filename}")
# async def download_image(filename: str):
#     file_path = f"download/{filename}"
#     if os.path.exists(file_path):
#         return FileResponse(file_path, media_type="image/jpeg")
#     else:
#         return {"error": "File not found"}

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)


import os
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
import io
import argparse
from stable_diffusion_tf.stable_diffusion import StableDiffusion

app = FastAPI()

@app.post("/generate_image")
async def generate_image(prompt: str, image: UploadFile = File(...)):
    # Save the uploaded image
    img = Image.open(io.BytesIO(await image.read()))
    input_image_path = "input_image.jpeg"
    img.save(input_image_path)

    # Generate the modified image
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, nargs="?", required=True, help="the prompt to render")
    parser.add_argument("--input", type=str, nargs="?", required=True, help="the input image filename")
    args = parser.parse_args([f"--prompt={prompt}", "--input=input_image.jpeg"])

    generator = StableDiffusion(
        img_height=512,
        img_width=512,
        jit_compile=False
    )

    modified_img = generator.generate(
        args.prompt,
        num_steps=50,
        unconditional_guidance_scale=7.5,
        temperature=1,
        batch_size=1,
        input_image=args.input,
        input_image_strength=0.8
    )

    # Save the modified image
    output_image_path = "download/output_image.jpeg"
    Image.fromarray(modified_img[0]).save(output_image_path)

    # Create a download link for the image
    output_filename = "output_image.jpeg"
    image_link = f"http://localhost:8000/download/{output_filename}"

    return {"link": image_link}

@app.get("/download/{filename}")
async def download_image(filename: str):
    file_path = f"download/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    else:
        return {"error": "File not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
