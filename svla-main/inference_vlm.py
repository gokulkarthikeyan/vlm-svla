# -------------------- Install missing packages --------------------

# -------------------- Imports --------------------
import os
import random
import torch
import nltk
import librosa
from PIL import Image
from datasets import load_dataset
from transformers import AutoTokenizer, Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from llava.model import LlavaQwen2ForCausalLM
from llava.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_AUDIO_START_TOKEN,
    DEFAULT_AUDIO_END_TOKEN,
    DEFAULT_AUDIO_TOKEN,
)
from melo.api import TTS
import evaluate

# -------------------- Fix for Melo TTS --------------------
nltk.download('averaged_perceptron_tagger_eng')

# -------------------- Device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- Load Dataset --------------------
# Using a small slice for demo (10 samples), expand later if GPU allows
dataset = load_dataset("speechcoco", split="train[:1%]")

# -------------------- Load SVLA Model --------------------
MODEL_PATH = "./weights/svla-sft-text-ins"  # make sure weights are uploaded to this path
model = LlavaQwen2ForCausalLM.from_pretrained(
    MODEL_PATH, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True
)
vision_tower = model.get_vision_tower()
vision_tower.load_model(device_map="auto")
image_processor = vision_tower.image_processor
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# -------------------- Load ASR Model --------------------
asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)

# -------------------- Metrics --------------------
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

# -------------------- TTS Model --------------------
tts_model = TTS(language='EN', device="cuda" if torch.cuda.is_available() else "cpu")
speaker_ids = tts_model.hps.data.spk2id
speakers = ['EN-US', 'EN-BR', 'EN_INDIA', 'EN-AU', 'EN-Default']

# -------------------- Evaluation Loop --------------------
all_preds = []
all_refs = []

for i, sample in enumerate(dataset.select(range(10))):  # demo on 10 samples
    # --- image ---
    image = Image.open(sample["image"]).convert("RGB")
    img_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(device)

    # --- audio ---
    speech_array, sr = librosa.load(sample["audio"]["path"], sr=16000)
    input_values = asr_tokenizer(speech_array, return_tensors="pt", padding="longest").input_values.to(device)

    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    audio_text = asr_tokenizer.decode(predicted_ids[0])

    # --- ground truth caption ---
    reference_caption = sample["captions"][0]["text"]

    # --- format prompt ---
    formatted_prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n"
        f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN*128}{DEFAULT_AUDIO_END_TOKEN}\n"
        f"Transcribed speech: {audio_text}\n"
        "Describe this image and audio.<|im_end|>\n<|im_start|>assistant\n"
    )

    input_ids = tokenizer([formatted_prompt], return_tensors="pt", add_special_tokens=False).to(device)

    # --- generate model output ---
    outputs = model.generate(
        inputs=input_ids["input_ids"],
        images=img_tensor,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # --- save speech output ---
    speaker = random.choice(speakers)
    tts_model.tts_to_file(response, speaker_ids[speaker], f"output_{i}.wav", speed=1.0)

    print(f"\nSample {i+1}")
    print("Audio Transcript:", audio_text)
    print("Reference Caption:", reference_caption)
    print("Model Response:", response)

    # collect for metrics
    all_preds.append(response)
    all_refs.append(reference_caption)

# -------------------- Compute Metrics --------------------
acc = accuracy.compute(predictions=all_preds, references=all_refs)
prec = precision.compute(predictions=all_preds, references=all_refs, average="macro")
rec = recall.compute(predictions=all_preds, references=all_refs, average="macro")
bleu_score = bleu.compute(predictions=[p.split() for p in all_preds],
                          references=[[r.split()] for r in all_refs])
rouge_score = rouge.compute(predictions=all_preds, references=all_refs)
meteor_score = meteor.compute(predictions=all_preds, references=all_refs)

print("\n==== Final Evaluation Metrics ====")
print("Accuracy:", acc["accuracy"])
print("Precision:", prec["precision"])
print("Recall:", rec["recall"])
print("Fluency (BLEU):", bleu_score["bleu"])
print("ROUGE-L:", rouge_score["rougeL"])
print("METEOR:", meteor_score["meteor"])
