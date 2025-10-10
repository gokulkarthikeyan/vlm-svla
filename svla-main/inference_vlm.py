from ipywidgets import VBox, HBox, Button, FileUpload, Text, Output, ButtonStyle
from IPython.display import display, clear_output
from PIL import Image
import io
import torch
from inference.audio_encoder import audio_encoder  # your existing audio processing
from inference.tokens_to_audio import decode_speech
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
import random

# --- Widgets ---
question_text = Text(value='', description='Question:', placeholder='Type your question here...')
image_upload = FileUpload(accept='image/*', description='Upload Image (optional)')
audio_upload = FileUpload(accept='audio/*', description='Upload Audio (optional)')
submit_button = Button(description='Submit', button_style='success')
reset_button = Button(description='Reset', button_style='warning')
output = Output()

# --- Model setup (load once) ---
MODEL_PATH = "./weights/svla-sft-text-ins"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaQwen2ForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True, device_map='auto', trust_remote_code=True)
vision_tower = model.get_vision_tower()
vision_tower.load_model(device_map="cuda:0")
image_processor = vision_tower.image_processor
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
text_to_audio_model = TTS(language='EN', device=device)
speaker_ids = text_to_audio_model.hps.data.spk2id

# --- Reset button functionality ---
def on_reset_clicked(b):
    question_text.value = ''
    image_upload.value.clear()
    audio_upload.value.clear()
    with output:
        clear_output()

reset_button.on_click(on_reset_clicked)

# --- Submit button functionality ---
def on_submit_clicked(b):
    with output:
        clear_output()
        # --- Get text input ---
        prompt = question_text.value.strip()
        if not prompt:
            print("Please type a question.")
            return
        
        # --- Process image if uploaded ---
        image_tensor = None
        if image_upload.value:
            try:
                uploaded_file = next(iter(image_upload.value.values()))
                img_data = uploaded_file['content']
                image = Image.open(io.BytesIO(img_data)).convert('RGB')
                image = image_processor(image, return_tensors='pt')["pixel_values"][0].to(device)
                image_tensor = image.unsqueeze(0).float()
            except Exception as e:
                print(f"Error processing image: {e}")
        
        # --- Process audio if uploaded ---
        if audio_upload.value:
            try:
                uploaded_file = next(iter(audio_upload.value.values()))
                audio_path = "temp_audio.wav"
                with open(audio_path, "wb") as f:
                    f.write(uploaded_file['content'])
                prompt = audio_encoder(audio_path)  # Convert audio to text prompt
            except Exception as e:
                print(f"Error processing audio: {e}")
        
        # --- Build final prompt ---
        system_prompt = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
        if image_tensor is not None:
            final_prompt = f"{system_prompt}\n<|im_start|>user\n<image>{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            final_prompt = f"{system_prompt}\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

        # --- Generate response ---
        try:
            input_ids = tokenizer([final_prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            outputs = model.generate(inputs=input_ids, images=image_tensor, max_new_tokens=512, do_sample=True)
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print("Response:\n", generated_text)
            
            # --- Optional: convert to speech ---
            if "||audio-" in generated_text:
                decode_speech(generated_text, device, "speech_answer.wav")
                print("Speech saved to 'speech_answer.wav'")
        except Exception as e:
            print(f"Error generating text: {e}")

submit_button.on_click(on_submit_clicked)

# --- Display UI ---
ui = VBox([
    question_text,
    image_upload,
    audio_upload,
    HBox([submit_button, reset_button]),
    output
])

display(ui)
