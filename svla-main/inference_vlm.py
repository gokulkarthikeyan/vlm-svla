import torch
from PIL import Image
import os
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from inference.tokens_to_audio import decode_speech

# ----------------------- CONFIG -----------------------
MODEL_PATH = "./weights/svla-sft-text-ins/svla-sft-text-ins"   # ✅ adjust path after unzip
IMAGE_PATH = "/kaggle/input/mydataset/dog.jpeg"                # ✅ your dataset image
PROMPT = "Summarize the picture."
# ------------------------------------------------------

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

def load_model_and_tokenizer(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(model_path, 
                                                  low_cpu_mem_usage=True, 
                                                  device_map='cuda', 
                                                  trust_remote_code=True)
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map="cuda:0")
    image_processor = vision_tower.image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, image_processor

def generate_text(model, tokenizer, image, prompt, max_new_tokens=512):
    if image is not None:
        image = image.unsqueeze(0).float().to("cuda:0")
    input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)["input_ids"]
    input_ids = input_ids.to("cuda:0")
    outputs = model.generate(
        inputs=input_ids,
        images=image,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=1.0,
        repetition_penalty=1.2,
        do_sample=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("Loading model...")
    model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH)

    print("Loading image...")
    image = resize_image_if_necessary(Image.open(IMAGE_PATH))
    image = image_processor(image, return_tensors='pt')["pixel_values"][0]

    print("Generating answer...")
    system = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
    formatted_prompt = f"{system}\n<|im_start|>user\n<image>\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"

    output = generate_text(model, tokenizer, image, formatted_prompt)
    print("\n===== MODEL OUTPUT =====\n")
    print(output)
    print("\n========================\n")

if __name__ == "__main__":
    main()
