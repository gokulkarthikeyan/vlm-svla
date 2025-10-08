import os
import torch
import librosa
import numpy as np
from PIL import Image
from llava.model import LlavaQwen2ForCausalLM
from inference.audio_encoder import audio_encoder
from melo.api import TTS
from inference.tokens_to_audio import decode_speech
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from jiwer import wer, cer  # For metrics

# ----------------- CONFIG -----------------
MODEL_PATH = "./weights/svla-sft-text-ins"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIBRISPEECH_PATH = "/kaggle/working/LibriSpeech/train-clean-100"
SPEECH_OUTPUT_PATH = "speech_question.wav"
SAMPLE_LIMIT = 20  # Limit number of samples for Kaggle runtime
IMAGE_PATH = None  # Optional: provide a single image for all samples, e.g., "./image.jpg"

# ----------------- UTILITY FUNCTIONS -----------------
def resize_image_if_necessary(image, longest_dimension=896):
    w, h = image.size
    if w <= longest_dimension and h <= longest_dimension:
        return image
    if w > h:
        new_w = longest_dimension
        new_h = int((longest_dimension / w) * h)
    else:
        new_h = longest_dimension
        new_w = int((longest_dimension / h) * w)
    return image.resize((new_w, new_h))

def load_model_and_tokenizer(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map='auto', trust_remote_code=True)
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map=DEVICE)
    tokenizer = model.get_tokenizer()
    return model, tokenizer, vision_tower.image_processor

def process_image(image_path, image_processor):
    image = Image.open(image_path)
    image = resize_image_if_necessary(image)
    return image_processor(image, return_tensors="pt")["pixel_values"][0].to(DEVICE)

def generate_text(model, tokenizer, image_tensor, prompt, max_new_tokens=256):
    input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEVICE)
    outputs = model.generate(
        inputs=input_ids,
        images=image_tensor.unsqueeze(0) if image_tensor is not None else None,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------- SVLA TTS + ASR -----------------
def svla_speech_asr_pipeline(text, tts_model, speaker_ids, asr_tokenizer, asr_model):
    # Generate speech from text using SVLA TTS
    tts_model.tts_to_file(text, speaker_ids["EN-US"], SPEECH_OUTPUT_PATH)
    # Encode speech to text (audio_encoder)
    encoded_prompt = audio_encoder(SPEECH_OUTPUT_PATH)
    
    # Optional: Decode speech output using Wav2Vec2 ASR
    audio, sr = librosa.load(SPEECH_OUTPUT_PATH, sr=16000)
    input_values = asr_tokenizer(audio, return_tensors="pt", padding="longest").input_values.to(DEVICE)
    logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_tokenizer.decode(predicted_ids[0]).upper()
    return transcription

# ----------------- EVALUATION -----------------
def evaluate_librispeech_svla(model, tokenizer, image_processor, image_tensor=None):
    print(f"Evaluating LibriSpeech using SVLA TTS → ASR pipeline...")
    
    # Load ASR model
    asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(DEVICE)
    
    # Load SVLA TTS
    tts_model = TTS(language='EN', device=DEVICE)
    speaker_ids = tts_model.hps.data.spk2id

    all_wers, all_cers = [], []
    count = 0

    for root, dirs, files in os.walk(LIBRISPEECH_PATH):
        for file in files:
            if file.endswith(".txt"):
                transcript_file = os.path.join(root, file)
                with open(transcript_file, "r") as f:
                    ground_truth = f.read().strip().upper()

                # Generate transcription via SVLA TTS → ASR
                try:
                    transcription = svla_speech_asr_pipeline(ground_truth, tts_model, speaker_ids, asr_tokenizer, asr_model)
                except Exception as e:
                    print(f"Error processing {transcript_file}: {e}")
                    continue

                all_wers.append(wer(ground_truth, transcription))
                all_cers.append(cer(ground_truth, transcription))

                count += 1
                if count >= SAMPLE_LIMIT:
                    break
        if count >= SAMPLE_LIMIT:
            break

    print(f"Evaluated {count} samples")
    print(f"Average WER: {np.mean(all_wers):.4f}")
    print(f"Average CER: {np.mean(all_cers):.4f}")

# ----------------- MAIN -----------------
def main():
    model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH)
    
    if IMAGE_PATH is not None and os.path.exists(IMAGE_PATH):
        image_tensor = process_image(IMAGE_PATH, image_processor)
    else:
        image_tensor = None
    
    evaluate_librispeech_svla(model, tokenizer, image_processor, image_tensor)

if __name__ == "__main__":
    main()
