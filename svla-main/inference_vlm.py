# =========================================================
# SVLA Multimodal Inference - Kaggle Non-Interactive Version
# (Preserves original structure/functions; removes input() loops)
# =========================================================

import os
import torch
from PIL import Image, UnidentifiedImageError
import requests
import librosa
import numpy as np
import soundfile as sf

# Hugging Face / Transformers
from transformers import AutoTokenizer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# LLaVA / SVLA
from llava.model import LlavaQwen2ForCausalLM, LlavaQwen2Config
from llava.constants import (
    IGNORE_INDEX, IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    DEFAULT_AUDIO_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_END_TOKEN
)

# TTS + audio utils
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from inference.tokens_to_audio import decode_speech

# ------------------------------------------------------------------
#                      KAGGLE CONFIG (EDIT THESE)
# ------------------------------------------------------------------

# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Model path
MODEL_PATH = "./weights/svla-sft-text-ins"

# Dataset base folder (Kaggle)
KAGGLE_INPUT_DIR = "/kaggle/input"
DATASET_NAME = "mydataset"   # change to your dataset folder name
DATASET_DIR = os.path.join(KAGGLE_INPUT_DIR, DATASET_NAME)

# Inputs: put either absolute Kaggle paths or URLs here.
# You can mix paths and URLs. Leave empty list to run with no image.
IMAGE_INPUTS = [
    # Examples:
    # "/kaggle/input/mydataset/dog.jpeg",
    # "https://example.com/sample.jpg",
]

# Optional: if you want to test a single image quickly:
SINGLE_IMAGE = ""   # e.g., "/kaggle/input/mydataset/dog.jpeg" or URL; if set, it overrides IMAGE_INPUTS

# Text prompts
DEFAULT_PROMPT = "What is happening in this image?"
EXTRA_PROMPT = ""  # e.g., "Answer in one sentence."

# TTS / ASR behavior
GENERATE_SPEECH_ANSWER = True   # if model returns audio token stream, decode to speech
RUN_ASR_ON_SPEECH_ANSWER = False  # if True, run ASR to print transcript
ASR_SAMPLE_RATE = 16000

# Speed for question TTS if you generate speech questions (disabled by default below)
TTS_SPEED = 1.0

# If you want to create an audio-question from text and feed via audio_encoder:
USE_SPOKEN_QUESTION = False
SPOKEN_QUESTION_TEXT = ""  # if non-empty and USE_SPOKEN_QUESTION=True, it will be encoded and used

# Output files
OUTPUT_SPOKEN_QUESTION_WAV = "speech_question.wav"
OUTPUT_SPOKEN_ANSWER_WAV = "speech_answer.wav"

# ------------------------------------------------------------------
#                 ORIGINAL HELPERS (KEPT & ADAPTED)
# ------------------------------------------------------------------

# (Kept constant from your code; note: not used directly in this Kaggle version)
IMAGE_TOKEN_INDEX = -200

system = "System: You serve as a language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # Kept from your original code (not required for the non-interactive path)
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

def resize_image_if_necessary(image: Image.Image, longest_dimension: int = 896) -> Image.Image:
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

def load_model_and_tokenizer(model_path):
    # Use single-GPU if available (cuda:0) to match Kaggle
    model = LlavaQwen2ForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True
    )
    model = model.to(DEVICE)
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map=DEVICE)
    image_processor = vision_tower.image_processor
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer, image_processor

def process_image(image, image_processor):
    try:
        processor_output = image_processor.preprocess(image, return_tensors="pt")
        # Move to DEVICE
        for k in processor_output:
            processor_output[k] = processor_output[k].to(DEVICE)
        return processor_output
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def _system_block():
    # Original meta-system adapted to Qwen chat format
    return "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"

def _format_prompt_with_or_without_image(user_prompt: str, has_image: bool) -> str:
    # Matches your original formatting: when image is used, inject 256 image tokens
    sysblk = _system_block()
    if has_image:
        return (
            f"{sysblk}\n"
            f"<|im_start|>user\n"
            f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n"
            f"{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    else:
        return (
            f"{sysblk}\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

def generate_text(model, tokenizer, image_pixel_values, prompt,
                  max_new_tokens=1024, temperature=0.7, top_p=1.0, repetition_penalty=1.3):
    """
    Fixed: call model.generate with input_ids (not 'inputs'), and pass 'images='
    when image_pixel_values is provided. Keeps your original defaults.
    """
    try:
        enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"].to(DEVICE)

        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=(temperature > 0.0),
        )
        if image_pixel_values is not None:
            # LLaVA/SVLA expects "images" kwarg with pixel_values [B,3,H,W]
            gen_kwargs["images"] = image_pixel_values

        with torch.no_grad():
            outputs = model.generate(**gen_kwargs)

        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Error generating text: {e}\n")
        return None

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True, timeout=15)
        response.raise_for_status()
        image = Image.open(response.raw).convert("RGB")
        image.load()
        return resize_image_if_necessary(image)
    except requests.RequestException as e:
        print(f"Error loading image: {e}\n")
        return None
    except UnidentifiedImageError:
        print("Error: The URL does not point to a valid image file.\n")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}\n")
        return None

def load_image_from_path(path):
    try:
        return resize_image_if_necessary(Image.open(path).convert("RGB"))
    except Exception as e:
        print(f"Error loading image from file path: {e}\n")
        return None

# ------------------------------------------------------------------
#                   KAGGLE-NATIVE NON-INTERACTIVE MAIN
# ------------------------------------------------------------------

def main():
    print("Loading model and tokenizer...\n")
    model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH)

    # Optional TTS and ASR init
    # Note: TTS only used for spoken Q; ASR only if RUN_ASR_ON_SPEECH_ANSWER
    text_to_audio_model = None
    speaker_ids = None
    if USE_SPOKEN_QUESTION or GENERATE_SPEECH_ANSWER:
        try:
            text_to_audio_model = TTS(language='EN', device=DEVICE)
            speaker_ids = text_to_audio_model.hps.data.spk2id
            print("[INFO] TTS initialized.")
        except Exception as e:
            print(f"[WARN] TTS init failed: {e}")

    asr_tokenizer = None
    asr_model = None
    if RUN_ASR_ON_SPEECH_ANSWER:
        try:
            asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
            asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
            asr_model = asr_model.to(DEVICE if "cuda" in DEVICE else "cpu")
            print("[INFO] ASR initialized.")
        except Exception as e:
            print(f"[WARN] ASR init failed: {e}")

    # Determine list of images to process
    to_process = []
    if SINGLE_IMAGE:
        to_process = [SINGLE_IMAGE]
    else:
        to_process = IMAGE_INPUTS[:]  # copy

    if not to_process:
        print("[INFO] No images provided. Running once without image context.")

    # Prepare the final prompt text (default + extra)
    user_text_prompt = f"{DEFAULT_PROMPT} {EXTRA_PROMPT}".strip()
    # If using a spoken question, synthesize and encode as audio token string
    if USE_SPOKEN_QUESTION and SPOKEN_QUESTION_TEXT:
        if text_to_audio_model is not None and speaker_ids is not None:
            try:
                speaker = 'EN-Default' if 'EN-Default' in speaker_ids else list(speaker_ids.keys())[0]
                print(f"[INFO] Creating spoken question → {OUTPUT_SPOKEN_QUESTION_WAV}")
                text_to_audio_model.tts_to_file(
                    SPOKEN_QUESTION_TEXT,
                    speaker_ids[speaker],
                    OUTPUT_SPOKEN_QUESTION_WAV,
                    speed=TTS_SPEED
                )
                user_text_prompt = audio_encoder(OUTPUT_SPOKEN_QUESTION_WAV)
                print("[INFO] Spoken question encoded as audio tokens.")
            except Exception as e:
                print(f"[WARN] Failed spoken question path, using text prompt. Reason: {e}")

    # If no images: run once text-only
    if not to_process:
        image_pixel_values = None
        formatted_prompt = _format_prompt_with_or_without_image(user_text_prompt, has_image=False)
        generated_text = generate_text(model, tokenizer, image_pixel_values, formatted_prompt)
        print("************************************************* INPUT *************************************************\n")
        print(formatted_prompt)
        print("\n" + "-"*50 + "\n")
        print("************************************************* OUTPUT *************************************************\n")
        print(generated_text)
        print("\n" + "-"*50 + "\n")
        if GENERATE_SPEECH_ANSWER and generated_text and "||audio-" in generated_text:
            try:
                clean = generated_text.replace(".", "")
                decode_speech(clean, DEVICE, OUTPUT_SPOKEN_ANSWER_WAV)
                print(f"[INFO] Saved speech answer to '{OUTPUT_SPOKEN_ANSWER_WAV}'")
                if RUN_ASR_ON_SPEECH_ANSWER and asr_tokenizer and asr_model:
                    audio, _ = librosa.load(OUTPUT_SPOKEN_ANSWER_WAV, sr=ASR_SAMPLE_RATE)
                    input_values = asr_tokenizer(audio, return_tensors="pt", padding="longest", sampling_rate=ASR_SAMPLE_RATE).input_values.to(asr_model.device)
                    with torch.no_grad():
                        logits = asr_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = asr_tokenizer.decode(predicted_ids[0])
                    print(f"[ASR] {transcription}")
            except Exception as e:
                print(f"[WARN] Speech decode failed: {e}")
        print("Done.")
        return

    # Otherwise process each provided image (paths or URLs)
    for idx, input_source in enumerate(to_process, 1):
        print(f"\n[INFO] ({idx}/{len(to_process)}) Processing: {input_source}")
        image = None
        if input_source.startswith("http"):
            image = load_image_from_url(input_source)
        else:
            # If not absolute, allow relative path under dataset dir
            cand = input_source
            if not os.path.isabs(cand):
                cand = os.path.join(DATASET_DIR, input_source)
            if not os.path.exists(cand):
                print(f"[WARN] File not found: {cand}")
            else:
                image = load_image_from_path(cand)

        has_image = image is not None
        image_pixel_values = None
        if has_image:
            # image_processor(..., return_tensors="pt") returns dict with pixel_values
            try:
                image_pixel_values = image_processor(image, return_tensors='pt')["pixel_values"].to(DEVICE)
            except Exception as e:
                print(f"[WARN] Image processing failed: {e}")
                has_image = False
                image_pixel_values = None

        formatted_prompt = _format_prompt_with_or_without_image(user_text_prompt, has_image=has_image)
        generated_text = generate_text(model, tokenizer, image_pixel_values, formatted_prompt)

        print("************************************************* INPUT *************************************************\n")
        print(formatted_prompt)
        print("\n" + "-"*50 + "\n")
        print("************************************************* OUTPUT *************************************************\n")
        print(generated_text)
        print("\n" + "-"*50 + "\n")

        if GENERATE_SPEECH_ANSWER and generated_text and "||audio-" in generated_text:
            try:
                clean = generated_text.replace(".", "")
                decode_speech(clean, DEVICE, OUTPUT_SPOKEN_ANSWER_WAV)
                print(f"[INFO] Saved speech answer to '{OUTPUT_SPOKEN_ANSWER_WAV}'")
                if RUN_ASR_ON_SPEECH_ANSWER and asr_tokenizer and asr_model:
                    audio, _ = librosa.load(OUTPUT_SPOKEN_ANSWER_WAV, sr=ASR_SAMPLE_RATE)
                    input_values = asr_tokenizer(audio, return_tensors="pt", padding="longest", sampling_rate=ASR_SAMPLE_RATE).input_values.to(asr_model.device)
                    with torch.no_grad():
                        logits = asr_model(input_values).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = asr_tokenizer.decode(predicted_ids[0])
                    print(f"[ASR] {transcription}")
            except Exception as e:
                print(f"[WARN] Speech decode failed: {e}")

    print("\nThank you for using the configurable image-based conversation generator!\n")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
