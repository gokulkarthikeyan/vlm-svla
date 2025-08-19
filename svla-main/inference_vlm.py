import torch
from PIL import Image
import requests
from llava.model import LlavaQwen2ForCausalLM
from transformers import AutoTokenizer
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from inference.tokens_to_audio import decode_speech

# ----------------------- CONFIG -----------------------
MODEL_PATH = "./weights/svla-sft-text-ins"
IMAGE_PATH = "/kaggle/input/mydataset/dog.jpeg"  # Can also be a URL
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SPEECH_OUTPUT_PATH = "speech_answer.wav"
PROMPT_TEXT = "Describe the image in detail and explain what is happening. Include objects, animals, environment, colors, and possible context."
# ------------------------------------------------------

# Resize image if needed
def resize_image_if_necessary(image):
    w, h = image.size
    max_dim = 896
    if w <= max_dim and h <= max_dim:
        return image
    if w > h:
        new_w = max_dim
        new_h = int(max_dim * h / w)
    else:
        new_h = max_dim
        new_w = int(max_dim * w / h)
    return image.resize((new_w, new_h))

# Load image
def load_image(path_or_url):
    if path_or_url.startswith("http"):
        response = requests.get(path_or_url, stream=True)
        image = Image.open(response.raw).convert("RGB")
    else:
        image = Image.open(path_or_url).convert("RGB")
    return resize_image_if_necessary(image)

# Load model and tokenizer
def load_model_and_tokenizer(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, device_map=DEVICE, trust_remote_code=True
    )
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = vision_tower.image_processor
    return model, tokenizer, image_processor

# Generate text from model
def generate_text(model, tokenizer, image_tensor, prompt_text):
    input_ids = tokenizer([prompt_text], return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    if image_tensor is not None:
        image_tensor = image_tensor.unsqueeze(0).float().to(DEVICE)
    outputs = model.generate(
        inputs=input_ids,
        images=image_tensor,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
        repetition_penalty=1.3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------- MAIN -----------------------
def main():
    print("Loading model and tokenizer...")
    model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH)

    print("Processing image...")
    image = load_image(IMAGE_PATH)
    image_tensor = image_processor(image, return_tensors='pt')["pixel_values"][0]

    system_prompt = (
        "<|im_start|>system\n"
        "You are a helpful AI assistant. Carefully analyze the image and provide a detailed, reasoned description. "
        "Include objects, animals, environment, colors, and context. Do not hallucinate.\n"
        "<|im_end|>"
    )
    formatted_prompt = f"{system_prompt}\n<|im_start|>user\n<image>{PROMPT_TEXT}<|im_end|>\n<|im_start|>assistant\n"

    print("Generating text output...")
    generated_text = generate_text(model, tokenizer, image_tensor, formatted_prompt)
    print("\n************************************************* OUTPUT *************************************************\n")
    print(generated_text)

    print("Converting text to speech...")
    semantic_codes = audio_encoder.get_code_from_text(generated_text)
    decode_speech(semantic_codes, DEVICE, SPEECH_OUTPUT_PATH)
    print(f"Speech saved to {SPEECH_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
