import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import io
import os
import torch
import random
import librosa

# ----------------------------
# 1️⃣ Load your models
# ----------------------------
from inference.audio_encoder import audio_encoder
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
from transformers import AutoTokenizer
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

MODEL_PATH = "./weights/svla-sft-text-ins"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LlavaQwen2ForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True, device_map=device, trust_remote_code=True)
vision_tower = model.get_vision_tower()
vision_tower.load_model(device_map=device)
image_processor = vision_tower.image_processor
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

text_to_audio_model = TTS(language='EN', device=device)
speaker_ids = text_to_audio_model.hps.data.spk2id

# ASR model (optional)
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)

# ----------------------------
# 2️⃣ Create Widgets
# ----------------------------
image_upload = widgets.FileUpload(accept='image/*', multiple=False, description="Upload Image (optional)")
audio_upload = widgets.FileUpload(accept='audio/*', multiple=False, description="Upload Audio (optional)")
text_input = widgets.Text(placeholder='Type your question here...', description='Question:')
submit_button = widgets.Button(description="Submit", button_style='success')
reset_button = widgets.Button(description="Reset", button_style='warning')
output_log = widgets.Output()

# ----------------------------
# 3️⃣ Define Button Callbacks
# ----------------------------
def on_submit(btn):
    with output_log:
        clear_output()
        print("Processing your input...")

        # Handle Image
        if image_upload.value:
            image_data = list(image_upload.value.values())[0]['content']
            image = Image.open(io.BytesIO(image_data))
            display(image)
            image_tensor = image_processor(image, return_tensors="pt")["pixel_values"][0].to(device)
        else:
            image_tensor = None
            print("No image uploaded.")

        # Handle Audio
        if audio_upload.value:
            audio_data = list(audio_upload.value.values())[0]['content']
            audio_path = "uploaded_audio.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_data)
            print(f"Audio saved as {audio_path}")
            prompt_text = audio_encoder(audio_path)
        else:
            prompt_text = ""

        # Handle Text
        question = text_input.value.strip()
        if question != "":
            prompt_text += " " + question

        if prompt_text == "":
            print("No question or audio provided!")
            return

        # Format Prompt
        system_text = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
        if image_tensor is not None:
            formatted_prompt = f"{system_text}\n<|im_start|>user\n{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        else:
            formatted_prompt = f"{system_text}\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

        # Generate Response
        input_ids = tokenizer([formatted_prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        if image_tensor is not None:
            image_tensor = image_tensor.unsqueeze(0)
        outputs = model.generate(inputs=input_ids, images=image_tensor, max_new_tokens=1024, temperature=0.7, top_p=1.0)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("************************************************* OUTPUT *************************************************")
        print(response)

def on_reset(btn):
    image_upload.value.clear()
    audio_upload.value.clear()
    text_input.value = ""
    with output_log:
        clear_output()
        print("All inputs and outputs have been reset.")

submit_button.on_click(on_submit)
reset_button.on_click(on_reset)

# ----------------------------
# 4️⃣ Display UI
# ----------------------------
ui = widgets.VBox([text_input, image_upload, audio_upload, widgets.HBox([submit_button, reset_button]), output_log])
display(ui)
