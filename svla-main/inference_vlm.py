import torch
from PIL import Image
import os
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from llava.constants import (
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    IMAGE_TOKEN_INDEX
)
from inference.tokens_to_audio import decode_speech

# Constants
MODEL_PATH = "./weights/svla-sft-text-ins"
IMAGE_PATH = "/kaggle/input/mydataset/dog.jpeg"
speech_output_path = "speech_answer.wav"
speed = 1.0

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
def load_model_and_tokenizer(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, device_map=device, trust_remote_code=True
    )
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map=device)
    image_processor = vision_tower.image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, image_processor

# Resize image if necessary
def resize_image_if_necessary(image):
    original_width, original_height = image.size
    longest_dimension = 896
    if original_width <= longest_dimension and original_height <= longest_dimension:
        return image
    if original_width > original_height:
        new_width = longest_dimension
        new_height = int((longest_dimension / original_width) * original_height)
    else:
        new_height = longest_dimension
        new_width = int((longest_dimension / original_height) * original_width)
    return image.resize((new_width, new_height))

# Load image from path
def load_image_from_path(path):
    try:
        return resize_image_if_necessary(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"Error loading image from file path: {e}")
        return None

# Generate text
def generate_text(model, tokenizer, image_tensor, prompt, max_new_tokens=1024):
    if image_tensor is not None:
        image_tensor = image_tensor.unsqueeze(0).float().to(device)
    input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = input_ids.to(device)
    outputs = model.generate(
        inputs=input_ids,
        images=image_tensor,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=1.0,
        repetition_penalty=1.3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main execution
def main():
    model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH)
    text_to_audio_model = TTS(language='EN', device=device)
    
    print("************************************************* IM READY! *************************************************\n")

    # Load and process image
    image = load_image_from_path(IMAGE_PATH)
    if image is None:
        print("Failed to load image. Exiting...")
        return
    image_tensor = image_processor(image, return_tensors='pt')["pixel_values"][0]

    # Prepare prompt
    system = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
    prompt_text = f"{system}\n<|im_start|>user\n{DEFAULT_IM_START_TOKEN}{IMAGE_TOKEN_INDEX*256}{DEFAULT_IM_END_TOKEN}\nSummarize this image.<|im_end|>\n<|im_start|>assistant\n"

    # Generate summary text
    generated_text = generate_text(model, tokenizer, image_tensor, prompt_text)
    print("************************************************* OUTPUT *************************************************\n")
    print(generated_text)

    # Convert generated text to speech
    decode_speech(generated_text, device, speech_output_path)
    print(f"Speech saved to {speech_output_path}")

if __name__ == "__main__":
    main()
