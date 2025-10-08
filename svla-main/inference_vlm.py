import os
import torch
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

# -----------------------------
# Load pretrained ASR model
# -----------------------------
asr_model_name = "facebook/wav2vec2-large-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(asr_model_name)
model = Wav2Vec2ForCTC.from_pretrained(asr_model_name).to("cuda")
model.eval()

# -----------------------------
# Load LibriSpeech datasets
# -----------------------------
local_dataset_path = "/kaggle/working/LibriSpeech"  # adjust path if needed

train_dataset = load_dataset("librispeech_asr", "clean", split="train.100", cache_dir=local_dataset_path)
test_dataset = load_dataset("librispeech_asr", "clean", split="test.clean", cache_dir=local_dataset_path)

# -----------------------------
# Load metrics
# -----------------------------
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

# -----------------------------
# Helper function to transcribe audio
# -----------------------------
def transcribe(batch):
    audio = batch["audio"]["array"]
    input_values = tokenizer(audio, return_tensors="pt", sampling_rate=16000).input_values.to("cuda")
    with torch.no_grad():
        logits = model(input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(pred_ids)[0]
    batch["pred_text"] = transcription
    return batch

# -----------------------------
# Transcribe datasets
# -----------------------------
print("Transcribing train dataset...")
train_dataset = train_dataset.map(transcribe)

print("Transcribing test dataset...")
test_dataset = test_dataset.map(transcribe)

# -----------------------------
# Compute metrics
# -----------------------------
def compute_metrics(dataset):
    references = [x["text"] for x in dataset]
    predictions = [x["pred_text"] for x in dataset]
    wer_score = wer_metric.compute(predictions=predictions, references=references)
    cer_score = cer_metric.compute(predictions=predictions, references=references)
    return wer_score, cer_score

train_wer, train_cer = compute_metrics(train_dataset)
test_wer, test_cer = compute_metrics(test_dataset)

# -----------------------------
# Print metrics
# -----------------------------
print("=== LibriSpeech ASR Evaluation ===")
print(f"Train-clean-100: WER={train_wer:.4f}, CER={train_cer:.4f}")
print(f"Test-clean:       WER={test_wer:.4f}, CER={test_cer:.4f}")
