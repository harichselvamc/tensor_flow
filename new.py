# # import uvicorn
# # from fastapi import FastAPI, File, UploadFile
# # from PIL import Image
# # import io
# # import argparse
# # from stable_diffusion_tf.stable_diffusion import StableDiffusion

# # app = FastAPI()

# # @app.post("/generate_image")
# # async def generate_image(prompt: str, image: UploadFile = File(...)):
# #     # Save the uploaded image
# #     img = Image.open(io.BytesIO(await image.read()))
# #     img.save("input_image.jpeg")

# #     # Generate the modified image
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--prompt", type=str, nargs="?", required=True, help="the prompt to render")
# #     parser.add_argument("--input", type=str, nargs="?", required=True, help="the input image filename")
# #     args = parser.parse_args([f"--prompt={prompt}", "--input=input_image.jpeg"])

# #     generator = StableDiffusion(
# #         img_height=512,
# #         img_width=512,
# #         jit_compile=False
# #     )

# #     modified_img = generator.generate(
# #         args.prompt,
# #         num_steps=50,
# #         unconditional_guidance_scale=7.5,
# #         temperature=1,
# #         batch_size=1,
# #         input_image=args.input,
# #         input_image_strength=0.8
# #     )

# #     # Save the modified image
# #     output_image_path = "output_image.jpeg"
# #     Image.fromarray(modified_img[0]).save(output_image_path)

# #     return {"result": output_image_path}

# # if __name__ == "__main__":
# #     uvicorn.run(app, host="0.0.0.0", port=8000)


import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import argparse
from stable_diffusion_tf.stable_diffusion import StableDiffusion

app = FastAPI()

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
        num_steps=50,
        unconditional_guidance_scale=7.5,
        temperature=1,
        batch_size=1,
        input_image=args.input,
        input_image_strength=0.8
    )

    # Save the modified image
    output_image_path = "output_image.jpeg"
    Image.fromarray(modified_img[0]).save(output_image_path)

    return {"result": output_image_path}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)









# import uvicorn
# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# import io
# import argparse
# from stable_diffusion_tf.stable_diffusion import StableDiffusion

# app = FastAPI()

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
#         jit_compile=False,
#         model_path="diffusion_model.h5"  # Set the model path here
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

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)









# import uvicorn
# from fastapi import FastAPI, File, UploadFile
# from PIL import Image
# import io
# import argparse
# from stable_diffusion_tf.stable_diffusion import StableDiffusion

# app = FastAPI()

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
#     generator.load_model("diffusion_model.h5")  # Load the model from the provided path

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

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
