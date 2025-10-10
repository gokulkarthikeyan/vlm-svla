# =====================================================
# IMPORTS
# =====================================================
import os
import torch
from PIL import Image
from io import BytesIO
from ipywidgets import FileUpload, Text, Button, VBox, HBox, Layout
from IPython.display import display, Audio, clear_output
import librosa
import numpy as np
from melo.api import TTS
from inference.audio_encoder import audio_encoder  # Your speech-to-text function
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "./weights/svla-sft-text-ins"  # Path to your SVLA model
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
speech_output_path = "speech_answer.wav"

# =====================================================
# LOAD MODELS
# =====================================================
from llava.model import LlavaQwen2ForCausalLM
from transformers import AutoTokenizer

def load_svla_model(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(model_path, device_map="cuda", low_cpu_mem_usage=True, trust_remote_code=True)
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map="cuda")
    image_processor = vision_tower.image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, image_processor

model, tokenizer, image_processor = load_svla_model(MODEL_PATH)
tts_model = TTS(language="EN", device=DEVICE)

# =====================================================
# HELPER FUNCTIONS
# =====================================================
uploaded_image = None
uploaded_prompt = None

def resize_image_if_needed(image: Image.Image, longest_dim=896):
    w, h = image.size
    if w <= longest_dim and h <= longest_dim:
        return image
    if w > h:
        new_w = longest_dim
        new_h = int(longest_dim / w * h)
    else:
        new_h = longest_dim
        new_w = int(longest_dim / h * w)
    return image.resize((new_w, new_h))

def handle_image_upload(change):
    clear_output(wait=True)
    uploaded_file = image_upload.value
    if uploaded_file:
        for fname, file_info in uploaded_file.items():
            img = Image.open(BytesIO(file_info["content"]))
            img = resize_image_if_needed(img)
            display(img)
            global uploaded_image
            uploaded_image = img
    display(ui)

def handle_audio_upload(change):
    clear_output(wait=True)
    uploaded_file = audio_upload.value
    if uploaded_file:
        for fname, file_info in uploaded_file.items():
            with open("speech_question.wav", "wb") as f:
                f.write(file_info["content"])
            transcription = audio_encoder("speech_question.wav")
            print("Transcribed Audio:", transcription)
            global uploaded_prompt
            uploaded_prompt = transcription
    display(ui)

def text_to_speech(text, path="speech_answer.wav"):
    speaker = list(tts_model.hps.data.spk2id.keys())[0]
    tts_model.tts_to_file(text, tts_model.hps.data.spk2id[speaker], path)
    return path

def generate_text(prompt, image=None):
    system_prompt = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
    if image is not None:
        image_tensor = image_processor(image, return_tensors="pt")["pixel_values"][0].unsqueeze(0).to(DEVICE)
        formatted_prompt = f"{system_prompt}\n<|im_start|>user\n{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer([formatted_prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        outputs = model.generate(inputs=input_ids, images=image_tensor, max_new_tokens=1024, do_sample=True)
    else:
        formatted_prompt = f"{system_prompt}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = tokenizer([formatted_prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
        outputs = model.generate(inputs=input_ids, max_new_tokens=1024, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def handle_submit(btn):
    clear_output(wait=True)
    prompt_text = text_input.value if text_input.value else uploaded_prompt if 'uploaded_prompt' in globals() else ""
    if prompt_text == "":
        print("Please provide a question via text or audio.")
        display(ui)
        return
    answer = generate_text(prompt_text, image=uploaded_image if 'uploaded_image' in globals() else None)
    print("Prompt:", prompt_text)
    print("Answer:", answer)
    speech_path = text_to_speech(answer)
    display(Audio(speech_path))
    display(ui)

# =====================================================
# WIDGETS
# =====================================================
image_upload = FileUpload(accept="image/*", multiple=False)
image_upload.observe(handle_image_upload, names="value")

audio_upload = FileUpload(accept="audio/*", multiple=False)
audio_upload.observe(handle_audio_upload, names="value")

text_input = Text(value="", placeholder="Type your question here...", description="Question:")
submit_btn = Button(description="Submit", button_style="success")
submit_btn.on_click(handle_submit)

ui = VBox([
    HBox([image_upload, audio_upload]),
    text_input,
    submit_btn
])

display(ui)
