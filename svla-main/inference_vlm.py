import torch
from PIL import Image
import os
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from inference.tokens_to_audio import decode_speech

# Constants
MODEL_PATH = "./weights/svla-sft-text-ins"  # Update if your model path differs
IMAGE_PATH = "/kaggle/input/mydataset/dog.jpeg"  # Update with your image
SPEECH_OUTPUT_PATH = "speech_answer.wav"

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
def load_model_and_tokenizer(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map=device,
        trust_remote_code=True
    )
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map=device)
    image_processor = vision_tower.image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, image_processor

# Resize image if necessary
def resize_image_if_necessary(image, max_dim=896):
    w, h = image.size
    if w <= max_dim and h <= max_dim:
        return image
    if w > h:
        new_w = max_dim
        new_h = int(max_dim * h / w)
    else:
        new_h = max_dim
        new_w = int(max_dim * w / h)
    return image.resize((new_w, new_h))

# Load image from path
def load_image(path):
    try:
        return resize_image_if_necessary(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

# Generate text from model
def generate_text(model, tokenizer, image_tensor, prompt, max_new_tokens=1024):
    input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if image_tensor is not None:
        image_tensor = image_tensor.unsqueeze(0).float().to(device)
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
    # Load model, tokenizer, and image processor
    model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH)
    text_to_audio_model = TTS(language='EN', device=device)

    print("************************************************* IM READY! *************************************************\n")

    # Load and process image
    image = load_image(IMAGE_PATH)
    if image is None:
        print("Failed to load image. Exiting...")
        return
    image_tensor = image_processor(image, return_tensors='pt')["pixel_values"][0]

    # Prepare prompt
    system_prompt = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
    prompt_text = (
        f"{system_prompt}\n"
        "<|im_start|>user\n"
        "Summarize the image.\n"
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    # Generate text
    generated_text = generate_text(model, tokenizer, image_tensor, prompt_text)
    print("************************************************* OUTPUT *************************************************\n")
    print(generated_text)

    # Convert text to speech
    decode_speech(generated_text, device, SPEECH_OUTPUT_PATH)
    print(f"Speech saved to {SPEECH_OUTPUT_PATH}")

if __name__ == "__main__":
    main()
