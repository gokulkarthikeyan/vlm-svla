# =====================================================
# IMPORTS
# =====================================================
import os
import torch
import librosa
from PIL import Image
from transformers import (
    Wav2Vec2Tokenizer, Wav2Vec2ForCTC,
    AutoTokenizer
)
from ipywidgets import FileUpload
from io import BytesIO

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "./weights/svla-sft-text-ins"  # VQA model path
SAMPLE_RATE = 16000

# =====================================================
# HELPER FUNCTIONS
# =====================================================
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

def load_image_from_bytes(uploaded_file):
    try:
        image = Image.open(BytesIO(uploaded_file))
        return resize_image_if_necessary(image)
    except Exception as e:
        print(f"Error loading image: {e}")
        return None

def load_audio_from_bytes(uploaded_file):
    try:
        audio, sr = librosa.load(BytesIO(uploaded_file), sr=SAMPLE_RATE)
        return audio
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None

def load_model_and_tokenizer(model_path):
    from llava.model import LlavaQwen2ForCausalLM
    model = LlavaQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map="cuda")
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map="cuda:0")
    image_processor = vision_tower.image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, image_processor

# =====================================================
# LOAD MODELS
# =====================================================
print("Loading VQA model...")
model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH)
print("Loading Wav2Vec2 ASR model...")
asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")

# =====================================================
# UPLOAD IMAGE ONCE
# =====================================================
print("Upload your image (optional):")
image_upload = FileUpload(accept="image/*", multiple=False)
display(image_upload)

image_tensor = None
while image_upload.value == {}:
    pass  # wait until user uploads

for name, file_info in image_upload.value.items():
    image = load_image_from_bytes(file_info['content'])
    if image:
        image_tensor = image_processor(image, return_tensors="pt")["pixel_values"][0].to("cuda")
        print(f"✅ Image '{name}' loaded successfully.")
    break

# =====================================================
# INTERACTIVE PROMPT LOOP
# =====================================================
system_prompt = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"

while True:
    print("\n--- New Prompt ---")
    print("Type your text prompt OR upload audio (optional).")
    print("Commands: 'quit' to exit.\n")

    # Audio upload
    audio_upload = FileUpload(accept=".wav", multiple=False)
    display(audio_upload)
    
    text_prompt = input("Type your text prompt (leave blank if using audio): ")

    # Check quit
    if text_prompt.lower() == 'quit':
        break

    # Process audio if uploaded
    if audio_upload.value:
        for name, file_info in audio_upload.value.items():
            speech = load_audio_from_bytes(file_info['content'])
            if speech is not None:
                input_values = asr_tokenizer(speech, return_tensors="pt", padding="longest").input_values.to("cuda")
                logits = asr_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                audio_text = asr_tokenizer.decode(predicted_ids[0])
                print(f"✅ Audio '{name}' transcribed as: {audio_text}")
                if text_prompt:
                    text_prompt += " " + audio_text
                else:
                    text_prompt = audio_text
            break

    if not text_prompt:
        print("No input detected. Skipping...")
        continue

    # Build full prompt
    if image_tensor is not None:
        full_prompt = f"{system_prompt}\n<|im_start|>user\n<image>{text_prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        full_prompt = f"{system_prompt}\n<|im_start|>user\n{text_prompt}<|im_end|>\n<|im_start|>assistant\n"

    input_ids = tokenizer([full_prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
    outputs = model.generate(inputs=input_ids, images=image_tensor, max_new_tokens=1024)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\n************************************************* OUTPUT *************************************************\n")
    print(generated_text)
