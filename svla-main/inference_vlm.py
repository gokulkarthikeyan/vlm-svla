# fixed_svla_cli.py
import os
import random
import json
from io import BytesIO

import torch
import numpy as np
from PIL import Image, UnidentifiedImageError
import requests
import librosa
import soundfile as sf

from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter

from transformers import AutoTokenizer, Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# llava and melo imports (make sure these packages are available in your env)
from llava.model import LlavaQwen2ForCausalLM, LlavaQwen2Config
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from inference.tokens_to_audio import decode_speech

from llava.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    DEFAULT_AUDIO_TOKEN, DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN
)

# ---- Config ----
MODEL_PATH = "./weights/svla-sft-text-ins"
SPEECH_OUTPUT_PATH = "speech_question.wav"
SPEECH_ANSWER_PATH = "speech_answer.wav"
SAMPLE_RATE = 16000
IMAGE_MAX_DIM = 896
IMAGE_TOKEN_INDEX = -200  # local sentinel (if you need)
speed = 1.0

# ---- Utility: enhanced prompt with history/completion ----
def get_enhanced_input(prompt_text=">>> ", history_file=".input_history", completer_words=None):
    completer = WordCompleter(completer_words) if completer_words else None
    try:
        user_input = prompt(
            prompt_text,
            history=FileHistory(history_file),
            auto_suggest=AutoSuggestFromHistory(),
            completer=completer
        )
        return user_input
    except (KeyboardInterrupt, EOFError):
        return None

# ---- Image utilities ----
def resize_image_if_necessary(image, longest_dimension=IMAGE_MAX_DIM):
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

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        image.load()
        return resize_image_if_necessary(image)
    except requests.RequestException as e:
        print(f"Error loading image from URL: {e}")
        return None
    except UnidentifiedImageError:
        print("Error: The URL does not point to a valid image file.")
        return None
    except Exception as e:
        print(f"Unexpected error loading image: {e}")
        return None

def load_image_from_path(path):
    try:
        image = Image.open(path).convert("RGB")
        return resize_image_if_necessary(image)
    except Exception as e:
        print(f"Error loading image from file path: {e}")
        return None

# ---- Tokenizer helper (unchanged logic but fixed small issues) ----
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

# ---- Model loading ----
def load_model_and_tokenizer(model_path, device):
    # Loads Llava Qwen2 model and tokenizer. Use low_cpu_mem_usage and device_map if available.
    try:
        # trust_remote_code=True is sometimes required for custom models
        model = LlavaQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
        vision_tower = model.get_vision_tower()
        # load vision tower model onto GPU 0 if available
        try:
            vision_tower.load_model(device_map={"": device})
        except Exception:
            # fallback: try default loading
            vision_tower.load_model(device_map="auto")
        image_processor = vision_tower.image_processor
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        return model, tokenizer, image_processor
    except Exception as e:
        print(f"Error loading model/tokenizer: {e}")
        raise

# ---- Text generation wrapper ----
def generate_text(model, tokenizer, image_tensor, prompt, device, max_new_tokens=1024, temperature=0.7, top_p=1.0, repetition_penalty=1.3):
    try:
        if image_tensor is not None:
            # image_tensor expected to be a torch tensor on correct device and shaped properly
            image_input = image_tensor.unsqueeze(0).float().to(device)
        else:
            image_input = None

        input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        outputs = model.generate(
            inputs=input_ids,
            images=image_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=temperature > 0.0,
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error during generation: {e}")
        return None

# ---- Main interactive loop ----
def main():
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load heavy models (wrap in try/except in case of errors)
    try:
        model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH, device)
    except Exception as e:
        print("Failed to load main model. Exiting.")
        return

    # TTS and ASR
    tts_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        text_to_audio_model = TTS(language='EN', device=tts_device)
        speaker_ids = text_to_audio_model.hps.data.spk2id
    except Exception as e:
        print(f"Warning: TTS initialization failed: {e}")
        text_to_audio_model = None
        speaker_ids = {}

    try:
        asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
        asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
    except Exception as e:
        print(f"Warning: ASR model failed to load: {e}")
        asr_tokenizer = None
        asr_model = None

    print("Loading model and tokenizer... Done.")
    print("************************************************* IM READY! *************************************************")

    while True:
        input_source = get_enhanced_input(prompt_text="Enter the URL or the file path of the image (or 'no image' or 'quit'): ")
        if input_source is None:
            continue
        if input_source.strip().lower() == 'quit':
            break

        image = None
        input_source = input_source.strip()
        if input_source.startswith('http'):
            print("Loading image from URL...")
            image = load_image_from_url(input_source)
            if image is None:
                # ask again
                continue
        elif os.path.exists(input_source):
            print("Loading image from file path...")
            image = load_image_from_path(input_source)
            if image is None:
                continue
        elif input_source.lower() == "no image":
            image = None
            print("Proceeding without an image.")
        else:
            print("Invalid input. Please enter a valid URL, file path, 'no image' or 'quit'.")
            continue

        # if we have a PIL image, process it with the model's image_processor into tensor
        image_tensor = None
        if image is not None:
            try:
                proc = image_processor(image, return_tensors='pt')
                # typical key is "pixel_values"; adapt if your processor uses another key
                image_tensor = proc["pixel_values"][0] if "pixel_values" in proc else proc[list(proc.keys())[0]][0]
                # keep image tensor on same device as model generation later by moving in generate_text
            except Exception as e:
                print(f"Error when processing image for model: {e}")
                image_tensor = None

        round_count = 0
        while True:
            prompt_text = get_enhanced_input("Enter the prompt ('audio' to input audio, 'end' to switch image, 'quit' to exit): ")
            if prompt_text is None:
                continue
            prompt_text = prompt_text.strip()
            if prompt_text == "":
                print("Invalid prompt; please type something.")
                continue
            if prompt_text.lower() == 'quit':
                return
            if prompt_text.lower() == 'end':
                break

            # handle inline TTS -> audio encoder flow
            if prompt_text.lower() in ['audio', 'speech']:
                if text_to_audio_model is None:
                    print("TTS engine isn't available in this environment.")
                    continue

                textual_speech = get_enhanced_input("Type your speech here ('end' to go back, 'quit' to exit): ")
                if textual_speech is None:
                    continue
                textual_speech = textual_speech.strip()
                if textual_speech.lower() == 'quit':
                    return
                if textual_speech.lower() == 'end':
                    continue
                if textual_speech == "":
                    print("Empty speech; skipping.")
                    continue

                print(f"saving speech question to '{SPEECH_OUTPUT_PATH}'")
                # choose a speaker if exists, else fallback to first available
                speaker = random.choice(list(speaker_ids.keys())) if speaker_ids else None
                try:
                    if speaker is not None:
                        text_to_audio_model.tts_to_file(textual_speech, speaker_ids[speaker], SPEECH_OUTPUT_PATH, speed=speed)
                    else:
                        # fallback: call TTS with default speaker if API allows
                        text_to_audio_model.tts_to_file(textual_speech, list(speaker_ids.values())[0], SPEECH_OUTPUT_PATH, speed=speed)
                except Exception as e:
                    print(f"TTS error: {e}")
                    continue

                # convert speech to prompt via audio_encoder
                try:
                    prompt_text = audio_encoder(SPEECH_OUTPUT_PATH)
                except Exception as e:
                    print(f"audio_encoder failed: {e}")
                    continue

            round_count += 1

            # Build system + user prompt text (keep tokens as required by model)
            system_block = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
            if image_tensor is not None:
                # The original code used DEFAULT_IM_START_TOKEN etc and a repeated DEFAULT_IMAGE_TOKEN*256.
                # Keep that pattern unless your model expects a different sentinel
                formatted_prompt = (
                    f"{system_block}\n<|im_start|>user\n"
                    f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN * 256}{DEFAULT_IM_END_TOKEN}\n"
                    f"{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
                )
            else:
                formatted_prompt = f"{system_block}\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"

            # Generate
            generated_text = generate_text(model, tokenizer, image_tensor, formatted_prompt, device=device)
            if not generated_text:
                print("No output from model.")
                continue

            print("************************************************* INPUT *************************************************")
            print(formatted_prompt)
            print("\n" + "-"*50 + "\n")
            print("************************************************* OUTPUT *************************************************")
            print(generated_text)
            print("\n" + "-"*50 + "\n")

            # If model produced an audio token pointer like "||audio-<...>" then decode it.
            # The original code checked for "||audio-" in generated text. Keep that behavior.
            if "||audio-" in generated_text:
                # naive sanitization
                generated_key = generated_text.replace(".", "").strip()
                print(f"saving speech answer to '{SPEECH_ANSWER_PATH}'")
                try:
                    decode_speech(generated_key, "cuda:0" if torch.cuda.is_available() else "cpu", SPEECH_ANSWER_PATH)
                except Exception as e:
                    print(f"Failed to decode speech tokens to file: {e}")
                    continue

                # ask user if they want ASR of the answer
                asr_choice = get_enhanced_input("Do you want to do ASR for the output speech (y/yes/n/no): ")
                if asr_choice is None:
                    continue
                if asr_choice.strip().lower() in ['y', 'yes']:
                    if asr_model is None or asr_tokenizer is None:
                        print("ASR model/tokenizer not available.")
                        continue
                    print("Doing ASR...")
                    try:
                        # load audio at SAMPLE_RATE
                        audio, rate = librosa.load(SPEECH_ANSWER_PATH, sr=SAMPLE_RATE)
                        input_values = asr_tokenizer(audio, return_tensors="pt", padding="longest").input_values.to(device)
                        logits = asr_model(input_values).logits
                        predicted_ids = torch.argmax(logits, dim=-1)
                        transcription = asr_tokenizer.batch_decode(predicted_ids)[0]
                        print(f"ANSWERING IN SPEECH (ASR): {transcription}")
                    except Exception as e:
                        print(f"ASR failed: {e}")
                        continue

    print("Thank you for using the configurable image-based conversation generator!")

if __name__ == "__main__":
    main()
