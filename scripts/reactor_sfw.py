from PIL import Image
import io
import logging
import os
import comfy.model_management as model_management
from reactor_download import download
from scripts.reactor_logger import logger

_PREDICTOR = None
_PREDICTOR_KEY = None

def ensure_nsfw_model(nsfwdet_model_path):
    """Download NSFW detection model if it doesn't exist"""
    downloaded = 0
    nd_urls = [
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/config.json",
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/model.safetensors",
        "https://huggingface.co/AdamCodd/vit-base-nsfw-detector/resolve/main/preprocessor_config.json",
    ]
    for model_url in nd_urls:
        model_name = os.path.basename(model_url)
        model_path = os.path.join(nsfwdet_model_path, model_name)
        if not os.path.isfile(model_path) or os.path.getsize(model_path) == 0:
            os.makedirs(nsfwdet_model_path, exist_ok=True)
            download(model_url, model_path, model_name)
        if os.path.isfile(model_path) and os.path.getsize(model_path) > 0:
            downloaded += 1
    return downloaded == 3

SCORE = 0.979

logging.getLogger("transformers").setLevel(logging.ERROR)

def clear_model_cache():
    global _PREDICTOR, _PREDICTOR_KEY
    _PREDICTOR = None
    _PREDICTOR_KEY = None

def nsfw_image(img_data, model_path: str):
    global _PREDICTOR, _PREDICTOR_KEY
    device = model_management.get_torch_device()
    key = (os.path.realpath(model_path), str(device))
    if _PREDICTOR is None or _PREDICTOR_KEY != key:
        clear_model_cache()
        logger.status("Ensuring NSFW detection model exists...")
        if not ensure_nsfw_model(model_path):
            return True
        from transformers import pipeline

        _PREDICTOR = pipeline("image-classification", model=model_path, device=device)
        _PREDICTOR_KEY = key
    with Image.open(io.BytesIO(img_data)) as img:
        result = _PREDICTOR(img)
        if result[0]["label"] == "nsfw" and result[0]["score"] > SCORE:
            logger.status(f'NSFW content detected with score={result[0]["score"]}, skipping...')
            return True
        return False
