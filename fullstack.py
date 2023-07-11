from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from PIL import Image
import os
import uuid
import rembg
import cv2
import numpy as np
import dlib
from io import BytesIO
import torch
import IPython.display as display
import uvicorn
import argparse
from stable_diffusion_tf.stable_diffusion import StableDiffusion
import io


app = FastAPI()

TEMP_FOLDER = "temp"
OUTPUT_FOLDER = "output"
DOWNLOAD_FOLDER = "download"

def create_folders():
    os.makedirs(TEMP_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

@app.get("/")
def root():
    instructions = {
        "removebackground": {
            "description": "This method removes the background from an image.",
            "usage": "POST /removebackground",
            "parameters": {
                "file": "Select the image file to upload."
            }
        },
        "resize": {
            "description": "This method resizes an image to a specified size.",
            "usage": "POST /resize",
            "parameters": {
                "file": "Select the image file to upload."
            }
        },
        "cartoon": {
            "description": "This method converts an image into a cartoon-like version.",
            "usage": "POST /cartoon",
            "parameters": {
                "file": "Select the image file to upload."
            }
        },
        "convert": {
            "description": "This method converts an image into an anime-style image.",
            "usage": "POST /convert",
            "parameters": {
                "file": "Select the image file to upload."
            }
        },
        "headbody": {
            "description": "This method performs face swapping between a head image and a body image.",
            "usage": "POST /headbody",
            "parameters": {
                "head": "Select the head image file to upload.",
                "body": "Select the body image file to upload."
            }
        },
        "generateimage": {
            "description": "This method generates a modified image based on a prompt.",
            "usage": "POST /generate_image",
            "parameters": {
                "prompt": "Enter the prompt for generating the modified image.",
                "image": "Select the input image file to upload."
            }
        }
    }
    return instructions

@app.post("/removebackground")
async def remove_background_endpoint(file: UploadFile = File(...)):
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

    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)

    # Save the downloaded image
    with open(download_path, "wb") as f:
        f.write(output_data)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}

def resize_image(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    return img

@app.post("/resize")
async def resize_image_endpoint(file: UploadFile = File(...)):
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = resize_image(file_path, (1080, 1080))

    output_filename = file.filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    cv2.imwrite(download_path, image)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}

def convert_to_cartoon(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

@app.post("/cartoon")
async def convert_to_cartoon_endpoint(file: UploadFile = File(...)):
    create_folders()

    file_path = os.path.join(TEMP_FOLDER, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    image = convert_to_cartoon(file_path)

    output_filename = file.filename
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, image)

    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    cv2.imwrite(download_path, image)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}

def save_image(image_bytes):
    save_folder = "download"
    os.makedirs(save_folder, exist_ok=True)
    file_name = f"converted_{len(os.listdir(save_folder)) + 1}.png"
    file_path = os.path.join(save_folder, file_name)

    im = Image.open(BytesIO(image_bytes)).convert("RGB")
    im_resized = im.resize((1080, 1080))
    im_resized.save(file_path, format="PNG")

    return file_name

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", device=device).eval()
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", device=device)
base_url = "http://localhost:8000"

@app.post('/convert')
async def convert_image(file: UploadFile = File(...)):
    bytes_in = await file.read()
    im_out = face2paint(model, Image.open(BytesIO(bytes_in)).convert("RGB"), side_by_side=False)
    buffer_out = BytesIO()
    im_out.save(buffer_out, format="PNG")
    bytes_out = buffer_out.getvalue()

    output_filename = save_image(bytes_out)
    image_link = f"{base_url}/download/{output_filename}"

    return {"link": image_link}

def swap_faces(head_image, body_image):
    detector = dlib.get_frontal_face_detector()
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    model_path = "dlib_face_recognition_resnet_model_v1.dat"
    facerec = dlib.face_recognition_model_v1(model_path)
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    head_gray = cv2.cvtColor(head_image, cv2.COLOR_BGR2GRAY)
    head_faces = detector(head_gray)

    if len(head_faces) == 0:
        return {"error": "No face detected in the head image."}

    head_face = head_faces[0]

    head_shape = predictor(head_gray, head_face)

    (head_x, head_y, head_w, head_h) = (head_face.left(), head_face.top(), head_face.width(), head_face.height())
    head_face_region = head_image[head_y:head_y + head_h, head_x:head_x + head_w]

    body_gray = cv2.cvtColor(body_image, cv2.COLOR_BGR2GRAY)

    body_faces = face_cascade.detectMultiScale(body_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(body_faces) == 0:
        return {"error": "No face detected in the body image."}

    (body_x, body_y, body_w, body_h) = body_faces[0]

    body_face_region = body_image[body_y:body_y + body_h, body_x:body_x + body_w]

    head_face_region_resized = cv2.resize(head_face_region, (body_w, body_h))

    head_mask = np.zeros(head_face_region_resized.shape[:2], dtype=np.uint8)
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    rect = (1, 1, head_face_region_resized.shape[1] - 1, head_face_region_resized.shape[0] - 1)
    cv2.grabCut(head_face_region_resized, head_mask, rect, bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)
    head_mask = np.where((head_mask == 2) | (head_mask == 0), 0, 1).astype('uint8')
    head_face_region_removed_bg = head_face_region_resized * head_mask[:, :, np.newaxis]

    body_image[body_y:body_y + body_h, body_x:body_x + body_w] = head_face_region_removed_bg

    return body_image

@app.post("/headbody")
async def swap_faces_endpoint(head: UploadFile = File(...), body: UploadFile = File(...)):
    create_folders()

    head_path = os.path.join(TEMP_FOLDER, head.filename)
    with open(head_path, "wb") as f:
        f.write(await head.read())

    body_path = os.path.join(TEMP_FOLDER, body.filename)
    with open(body_path, "wb") as f:
        f.write(await body.read())

    head_image = cv2.imread(head_path)
    body_image = cv2.imread(body_path)

    swapped_image = swap_faces(head_image, body_image)

    output_filename = f"headbody_{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)
    cv2.imwrite(output_path, swapped_image)

    download_path = os.path.join(DOWNLOAD_FOLDER, output_filename)
    cv2.imwrite(download_path, swapped_image)

    image_link = f"http://localhost:8000/download/{output_filename}"
    return {"link": image_link}

# @app.post("/generate_image")
# async def generate_image(prompt: str, image: UploadFile = File(...)):
#     # Save the uploaded image
#     img = Image.open(io.BytesIO(await image.read()))
#     img.save("input_image.jpeg")

#     # Generate the modified image
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--prompt", type=str, nargs="?", required=True, help="the prompt to render")
#     parser.add_argument("--input", type=str, nargs="?", required=True, help="the input image filename")
#     args = parser.parse_args([f"--prompt={prompt}", "--input=input_image.jpeg"])

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
#     output_image_path = "output_image.jpeg"
#     Image.fromarray(modified_img[0]).save(output_image_path)

#     return {"result": output_image_path}


@app.post("/generate_image")
async def generate_image(prompt: str, image: UploadFile = File(...)):
    # Save the uploaded image
    img = Image.open(io.BytesIO(await image.read()))
    img.save("input_image.jpeg")

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
