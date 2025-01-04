from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from diffusers import AutoPipelineForText2Image, AutoPipelineForImage2Image
import torch
from io import BytesIO
import time
import logging
import asyncio
from pydantic import BaseModel
from uuid import uuid4
import os
from PIL import Image
import numpy as np
from redis import Redis
from torchvision import transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

redis_client = Redis(
    host=os.environ.get("REDIS_HOST", "localhost"),
    port=os.environ.get("REDIS_PORT", 6379),
    db=0,
)


@app.middleware("http")
async def log_duration(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {duration:.2f} seconds")
    return response


# ----------------------
#  MODEL INITIALIZATION
# ----------------------
# If you have multiple GPUs, you can load multiple pipelines or specify an environment
# variable like GPU_ID. For now, we'll keep it single-GPU for simplicity and docker assumptions.
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# This can be any model with a text2img + image2image pipeline (e.g. stable diffusion)
model_name = os.environ.get("MODEL_NAME", "stabilityai/sdxl-turbo")

logger.info(f"Loading model {model_name} onto device: {device}")
txt2img_pipeline = AutoPipelineForText2Image.from_pretrained(
    model_name, torch_dtype=torch.float16, variant="fp16"
).to(device)

img2img_pipeline = AutoPipelineForImage2Image.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    variant="fp16",
    # Reuse components so they remain on the same device
    vae=txt2img_pipeline.vae,
    text_encoder=txt2img_pipeline.text_encoder,
    tokenizer=txt2img_pipeline.tokenizer,
    unet=txt2img_pipeline.unet,
    scheduler=txt2img_pipeline.scheduler,
    safety_checker=None,  # Turn off safety if you want
    feature_extractor=None,  # Turn off feature extractor if you want
).to(device)

# Lock for concurrency
gpu_lock = asyncio.Lock()
logger.info("GPU initialized and pipelines loaded")

# Where uploaded images (or other saved files) will go
IMAGE_DIR = "uploaded_images"
os.makedirs(IMAGE_DIR, exist_ok=True)


# ----------------------
#       DATA MODELS
# ----------------------
class Txt2ImgInput(BaseModel):
    prompt: str
    num_inference_steps: int = 4
    # You can expose more parameters as needed


class Img2ImgInput(BaseModel):
    prompt: str
    file_id: str
    num_inference_steps: int = 4
    strength: float = 0.5
    # Again, you can expose guidance_scale, scheduler, etc. if you want


# ----------------------
#        ENDPOINTS
# ----------------------


@app.post("/txt2img")
async def txt2img(input_data: Txt2ImgInput):
    """
    Simple text-to-image endpoint.
    Returns a PNG image as a streaming response with a random UUID as the filename.
    """
    try:
        async with gpu_lock:
            # We don't need gradients for inference
            with torch.no_grad():
                pipe = txt2img_pipeline
                # For typical stable diffusion, guidance_scale is often ~7.5,
                # but use whatever you prefer. Using 0.0 = no guidance.
                result = pipe(
                    prompt=input_data.prompt,
                    num_inference_steps=input_data.num_inference_steps,
                    guidance_scale=0.0,
                )
                image = result.images[0]

        # Convert image to in-memory bytes
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        # Generate a random UUID
        file_uuid = uuid4()
        filename = f"{file_uuid}.png"

        headers = {"Content-Disposition": f"attachment; filename={filename}"}
        return StreamingResponse(img_byte_arr, media_type="image/png", headers=headers)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload an image to Redis.
    Returns a file_id (UUID) so we can retrieve the image tensor later.
    """
    try:
        file_id = str("diffusor_data:", uuid4())

        # Read entire file content
        file_content = await file.read()
        # Store raw bytes in Redis
        redis_client.set(file_id, file_content)

        # Convert to PIL image
        pil_image = Image.open(BytesIO(file_content)).convert("RGB")

        # Convert to float16 [0,1], shape (H,W,3)
        np_image = np.array(pil_image) / 255.0
        np_image = np_image.astype(np.float16)

        # Convert to torch tensor shape [1,3,H,W]
        tensor_image = (
            torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0).to(torch.float16)
        )

        # Save that tensor in-memory
        tensor_bytes = BytesIO()
        torch.save(tensor_image, tensor_bytes)
        tensor_bytes.seek(0)

        # Store the tensor in Redis
        redis_client.set(f"{file_id}_tensor", tensor_bytes.getvalue())

        return {"file_id": file_id}
    except Exception as e:
        logger.error(f"Failed to upload image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/img2img")
async def img2img(input_data: Img2ImgInput):
    """
    Basic image-to-image endpoint.
    - Retrieves the stored tensor from Redis
    - Converts it back to a PIL image
    - Runs the image2image pipeline with the given prompt
    - Streams back the result as PNG
    """
    # Retrieve tensor data from Redis
    tensor_data = redis_client.get(f"diffusor_data:{input_data.file_id}_tensor")
    if not tensor_data:
        raise HTTPException(status_code=404, detail="File not found")

    # Convert bytes -> torch tensor
    tensor_bytes = BytesIO(tensor_data)
    tensor_bytes.seek(0)
    tensor_image = torch.load(tensor_bytes)

    # Convert the Torch tensor back to a PIL image
    # Typically shape is [1, 3, H, W], so let's remove batch dim.
    # We'll ensure it's CPU so that .numpy() works if needed.
    pil_image = T.ToPILImage()(tensor_image.squeeze(0).cpu())

    try:
        async with gpu_lock:
            with torch.no_grad():
                pipe = img2img_pipeline
                # Run inference:
                # - 'image=pil_image' since pipeline expects a PIL image or array
                # - You can adjust guidance_scale as you like; 0.0 means no guidance
                result = pipe(
                    prompt=input_data.prompt,
                    image=pil_image,
                    num_inference_steps=input_data.num_inference_steps,
                    strength=input_data.strength,
                    guidance_scale=0.0,
                )
                image = result.images[0]

        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)

        file_uuid = uuid4()
        filename = f"{file_uuid}.png"
        headers = {"Content-Disposition": f"attachment; filename={filename}"}
        return StreamingResponse(img_byte_arr, media_type="image/png", headers=headers)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
