import os
import torch
from PIL import Image
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from llava.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_AUDIO_TOKEN,
)
import librosa
from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from melo.api import TTS
import random

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- find image --------------------
img_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower() == "dog.jpeg":   # adjust name if needed
            img_path = os.path.join(root, f)
            break
if img_path:
    print("Using image:", img_path)
    image = Image.open(img_path).convert("RGB")
    # resize if too large
    max_dim = 896
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)))
else:
    image = None
    print("No image found, skipping image input.")

# -------------------- find audio --------------------
audio_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower().endswith((".wav", ".mp3", ".flac")):
            audio_path = os.path.join(root, f)
            break

if audio_path:
    print("Using audio:", audio_path)
else:
    print("No audio file found, skipping audio input.")

# -------------------- load SVLA model --------------------
MODEL_PATH = "./weights/svla-sft-text-ins"  # ensure weights are available
model = LlavaQwen2ForCausalLM.from_pretrained(
    MODEL_PATH, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True
)
vision_tower = model.get_vision_tower()
vision_tower.load_model(device_map="auto")
image_processor = vision_tower.image_processor
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# -------------------- load ASR model --------------------
if audio_path:
    asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)

    # load audio at 16k
    audio, sr = librosa.load(audio_path, sr=16000)
    input_values = asr_tokenizer(audio, return_tensors="pt", padding="longest").input_values.to(device)

    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    audio_text = asr_tokenizer.decode(predicted_ids[0])
    print("Transcribed Audio:", audio_text)
else:
    audio_text = ""

# -------------------- prepare prompt --------------------
prompt_text = (
    "Describe this image and audio." if (image and audio_text) else
    "Describe this image." if image else
    "Understand this audio." if audio_text else
    "Hello."
)

formatted_prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"

if image:
    formatted_prompt += f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n"
if audio_text:
    formatted_prompt += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN*128}{DEFAULT_AUDIO_END_TOKEN}\n"
    formatted_prompt += f"Transcribed speech: {audio_text}\n"

formatted_prompt += f"{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

# -------------------- generate --------------------
input_ids = tokenizer([formatted_prompt], return_tensors="pt", add_special_tokens=False).to(device)
img_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(device) if image else None

outputs = model.generate(
    inputs=input_ids["input_ids"],
    images=img_tensor,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print("\nOUTPUT (Text):\n", response)

# -------------------- text to speech --------------------
tts_model = TTS(language='EN', device="cuda" if torch.cuda.is_available() else "cpu")
speaker_ids = tts_model.hps.data.spk2id
speaker = random.choice(['EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU', 'EN-Default'])

output_audio_path = "model_output.wav"
tts_model.tts_to_file(response, speaker_ids[speaker], output_audio_path, speed=1.0)

print(f"OUTPUT (Speech saved at): {output_audio_path}")
