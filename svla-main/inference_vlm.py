import os
import torch
import librosa
import numpy as np
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer

# =====================================================
# CONFIGURATION
# =====================================================
LIBRISPEECH_PATH = "/kaggle/input/librispeeech/LibriSpeech"  # Path in your Kaggle dataset
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000

# =====================================================
# LOAD DATASET (LOCAL)
# =====================================================
# You can load only a subset for speed (train-clean-100)
def get_audio_files(path):
    audio_files = []
    texts = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".flac"):
                transcript_file = os.path.join(root, file.replace(".flac", ".txt"))
                if os.path.exists(transcript_file):
                    audio_files.append(os.path.join(root, file))
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        line = f.readline().strip()
                        # Example format: "84-121123-0000 THE TRANSCRIPTION"
                        texts.append(" ".join(line.split(" ")[1:]))
    return audio_files, texts

audio_files, texts = get_audio_files(LIBRISPEECH_PATH)
print(f"Loaded {len(audio_files)} audio files")

# Use only a smaller subset for training quickly
audio_files = audio_files[:200]
texts = texts[:200]

# =====================================================
# CREATE HUGGINGFACE DATASET
# =====================================================
data = Dataset.from_dict({"path": audio_files, "text": texts})
data = data.train_test_split(test_size=0.2)
print(data)

# =====================================================
# LOAD PROCESSOR AND MODEL
# =====================================================
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

# =====================================================
# PREPROCESS FUNCTION
# =====================================================
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=SAMPLE_RATE)
    batch["speech"] = speech_array
    batch["sampling_rate"] = sampling_rate
    batch["input_values"] = processor(speech_array, sampling_rate=SAMPLE_RATE).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

data = data.map(speech_file_to_array_fn, remove_columns=["path"])

# =====================================================
# TRAINING ARGUMENTS
# =====================================================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    save_total_limit=1,
    fp16=True,
    logging_dir="./logs",
)

# =====================================================
# METRICS
# =====================================================
wer_metric = load_metric("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# =====================================================
# TRAINER
# =====================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics,
)

# =====================================================
# TRAIN MODEL
# =====================================================
trainer.train()

# =====================================================
# EVALUATE
# =====================================================
metrics = trainer.evaluate()
print(metrics)
