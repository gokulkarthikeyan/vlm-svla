# =====================================================
# IMPORTS
# =====================================================
import torch
from PIL import Image
import librosa
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
import random
import os

# Your model imports
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from inference.tokens_to_audio import decode_speech
from llava.constants import (DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IM_END_TOKEN, DEFAULT_AUDIO_TOKEN,
                             DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN)

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "./weights/svla-sft-text-ins"
UPLOAD_DIR = Path("/kaggle/working/uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 16000
speed = 1.0

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def upload_file_widget(accept, description="Upload"):
    uploader = widgets.FileUpload(accept=accept, multiple=False, description=description)
    display(uploader)
    return uploader

def save_uploaded_file(uploader, save_dir=UPLOAD_DIR):
    if not uploader.value:
        return None
    uploaded_file = list(uploader.value.values())[0]
    file_path = save_dir / uploaded_file['metadata']['name']
    with open(file_path, 'wb') as f:
        f.write(uploaded_file['content'])
    return file_path

def load_image(path):
    try:
        return Image.open(path)
    except:
        return None

def load_audio(path, sr=SAMPLE_RATE):
    try:
        audio, _ = librosa.load(path, sr=sr)
        return audio
    except:
        return None

def resize_image_if_necessary(image, longest_dimension=896):
    original_width, original_height = image.size
    if original_width <= longest_dimension and original_height <= longest_dimension:
        return image
    if original_width > original_height:
        new_width = longest_dimension
        new_height = int((longest_dimension / original_width) * original_height)
    else:
        new_height = longest_dimension
        new_width = int((longest_dimension / original_height) * original_width)
    return image.resize((new_width, new_height))

# =====================================================
# LOAD MODEL & TOKENIZER
# =====================================================
print("Loading model, tokenizer, and TTS...")
model = LlavaQwen2ForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True)
vision_tower = model.get_vision_tower()
vision_tower.load_model(device_map="cuda:0")
image_processor = vision_tower.image_processor
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text_to_audio_model = TTS(language='EN', device="cuda:1")
speaker_ids = text_to_audio_model.hps.data.spk2id
asr_tokenizer = torch.hub.load('pytorch/fairseq', 'wav2vec2_large_lv60k')  # or your preferred tokenizer
asr_model = torch.hub.load('pytorch/fairseq', 'wav2vec2_large_lv60k')      # or your preferred ASR model

print("✅ Models loaded successfully!\n")

# =====================================================
# INTERACTIVE INPUT WIDGETS
# =====================================================
print("Step 1️⃣: Upload an image (optional)")
image_uploader = upload_file_widget(accept=".png,.jpg,.jpeg")
image_path = save_uploaded_file(image_uploader)
image = resize_image_if_necessary(load_image(image_path)) if image_path else None

print("Step 2️⃣: Upload an audio file (optional)")
audio_uploader = upload_file_widget(accept=".wav,.flac")
audio_path = save_uploaded_file(audio_uploader)
audio = load_audio(audio_path) if audio_path else None

question_widget = widgets.Text(
    value='',
    placeholder='Type your question here...',
    description='Question:',
    disabled=False
)
display(question_widget)

# =====================================================
# PROCESS INPUT AND GENERATE ANSWER
# =====================================================
def generate_vqa_answer(change):
    question = change['new']
    if not question.strip():
        print("❌ Empty question, type again!")
        return
    
    print("\nProcessing your input...\n")
    
    processed_image = None
    if image is not None:
        processed_image = image_processor(image, return_tensors='pt')["pixel_values"][0].unsqueeze(0).to("cuda:0")
    
    # Handle audio input
    if audio is not None:
        # Convert audio to prompt via your audio_encoder
        speech_prompt = audio_encoder(audio_path)
        final_prompt = f"{speech_prompt} {question}"
    else:
        final_prompt = question

    # Format prompt for model
    system_prompt = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
    if processed_image is not None:
        formatted_prompt = f"{system_prompt}\n<|im_start|>user\n{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n{final_prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        formatted_prompt = f"{system_prompt}\n<|im_start|>user\n{final_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate text answer
    input_ids = tokenizer([formatted_prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda:0")
    outputs = model.generate(inputs=input_ids, images=processed_image, max_new_tokens=1024)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("************************************************* INPUT *************************************************")
    print(formatted_prompt)
    print("\n************************************************* OUTPUT *************************************************")
    print(answer)
    
    # Optional: TTS output
    if "||audio-" in answer:
        decode_speech(answer.replace(".", ""), "cuda:0", "speech_answer.wav")
        print("Saved speech answer to 'speech_answer.wav'")
    
question_widget.observe(generate_vqa_answer, names='value')
