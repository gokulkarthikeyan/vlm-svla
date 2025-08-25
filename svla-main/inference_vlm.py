# =========================================================
# SVLA Multimodal Inference - Kaggle Non-Interactive Script
# =========================================================

import os
import torch
import librosa
import numpy as np
import soundfile as sf
from PIL import Image
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
from inference.audio_encoder import audio_encoder

# ----------------------- CONFIG -----------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "/kaggle/working/vlm-svla/svla-main/weights/svla-sft-text-ins"
IMAGE_PATH = "/kaggle/input/mydataset/dog.jpeg"
AUDIO_INPUT_PATH = "/kaggle/input/mydataset/sample.wav"  # optional
OUTPUT_SPEECH_PATH = "/kaggle/working/generated_response.wav"

# ----------------------- UTILS -----------------------
def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"[INFO] Loaded image: {image_path}")
        return image
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None

def load_audio(audio_path, target_sr=16000):
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)
        print(f"[INFO] Loaded audio: {audio_path} (sr={sr})")
        return audio, sr
    except Exception as e:
        print(f"[ERROR] Failed to load audio {audio_path}: {e}")
        return None, None

def save_audio(waveform, path, sr=16000):
    try:
        sf.write(path, waveform, sr)
        print(f"[INFO] Saved audio: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to save audio {path}: {e}")

# ----------------------- LOAD MODEL -----------------------
print("[STEP 1] Loading SVLA model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = LlavaQwen2ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    low_cpu_mem_usage=True
).to(DEVICE).eval()
print("[INFO] Model loaded successfully!")

# ----------------------- INPUTS -----------------------
print("[STEP 2] Preparing inputs...")

# Image
image = load_image(IMAGE_PATH)

# Audio (optional)
audio, sr = None, None
if os.path.exists(AUDIO_INPUT_PATH):
    audio, sr = load_audio(AUDIO_INPUT_PATH)

# Text prompt
text_prompt = "What do you see in this image?"
print(f"[INFO] Text prompt: {text_prompt}")

# ----------------------- INFERENCE -----------------------
print("[STEP 3] Running inference...")

# Build multimodal prompt
prompt = text_prompt
if image is not None:
    prompt = "<image>\n" + prompt
if audio is not None:
    prompt = "<audio>\n" + prompt

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(DEVICE)

# Encode audio into tokens if present
if audio is not None:
    audio_tokens = audio_encoder.get_code_from_wav(audio, sr)
    # insert tokens into inputs (model handles alignment)
    # depends on svla implementation, simplified here:
    inputs["audio_tokens"] = torch.tensor(audio_tokens).unsqueeze(0).to(DEVICE)

# Forward pass
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"[RESULT] Model output:\n{generated_text}")

# ----------------------- SPEECH OUTPUT -----------------------
print("[STEP 4] Converting text to speech...")
try:
    tts = TTS(language="EN", device=DEVICE)
    tts_out = tts.tts(generated_text)  # returns numpy waveform
    save_audio(tts_out, OUTPUT_SPEECH_PATH)
except Exception as e:
    print(f"[ERROR] TTS failed: {e}")

print("[DONE] Inference pipeline finished successfully!")
