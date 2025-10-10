# =====================================================
# IMPORTS
# =====================================================
import os
import torch
import librosa
import numpy as np
from datasets import Dataset
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import evaluate  # replaces deprecated load_metric

# =====================================================
# CONFIGURATION
# =====================================================
DATA_DIR = "/kaggle/input/librispeeech/LibriSpeech/train-clean-100"
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
MAX_FILES = 200  # reduce for faster training/evaluation

# =====================================================
# HELPER FUNCTION TO LOAD AUDIO + TRANSCRIPTS
# =====================================================
def load_librispeech_subset(path, max_files=None):
    audio_paths = []
    texts = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".flac"):
                transcript_file = os.path.join(root, file.replace(".flac", ".txt"))
                if os.path.exists(transcript_file):
                    with open(transcript_file, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split(" ", 1)
                            if len(parts) == 2 and parts[0] in file:
                                audio_paths.append(os.path.join(root, file))
                                texts.append(parts[1])
                                break
        if max_files and len(audio_paths) >= max_files:
            break

    return audio_paths[:max_files], texts[:max_files]

# =====================================================
# LOAD A SMALL SAMPLE OF LIBRISPEECH
# =====================================================
audio_files, transcripts = load_librispeech_subset(DATA_DIR, MAX_FILES)
print(f"✅ Loaded {len(audio_files)} audio files")

# =====================================================
# CREATE DATASET AND SPLIT
# =====================================================
dataset = Dataset.from_dict({"path": audio_files, "text": transcripts})
dataset = dataset.train_test_split(test_size=0.2)
print(dataset)

# =====================================================
# LOAD MODEL AND PROCESSOR
# =====================================================
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

# =====================================================
# FEATURE EXTRACTION FUNCTION
# =====================================================
def preprocess(batch):
    speech_array, _ = librosa.load(batch["path"], sr=SAMPLE_RATE)
    batch["input_values"] = processor(speech_array, sampling_rate=SAMPLE_RATE).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

dataset = dataset.map(preprocess, remove_columns=["path"], num_proc=2)

# =====================================================
# METRIC: WORD ERROR RATE (WER)
# =====================================================
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = np.argmax(pred.predictions, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# =====================================================
# TRAINING ARGUMENTS
# =====================================================
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="no",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    fp16=True,
    logging_dir="./logs",
    report_to="none"
)

# =====================================================
# TRAINER SETUP
# =====================================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

# =====================================================
# TRAIN MODEL
# =====================================================
trainer.train()

# =====================================================
# EVALUATE ON TEST SET
# =====================================================
metrics = trainer.evaluate()
print("✅ Final Evaluation Metrics:", metrics)
