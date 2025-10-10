# ================================
# IMPORTS
# ================================
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import torch
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)

# ================================
# CONFIGURATION
# ================================
DATASET_PATH = "/kaggle/input/librispeech"  # your uploaded dataset path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True if DEVICE == "cuda" else False
SAMPLE_RATE = 16000
BATCH_SIZE = 2          # adjust for CPU/GPU
EPOCHS = 3
LEARNING_RATE = 1e-4

# ================================
# STEP 1: LOAD FILES AND CREATE CSV
# ================================
# Assuming filenames like: HELLO_WORLD_001.wav
audio_files = [f for f in os.listdir(DATASET_PATH) if f.endswith(".wav")]

data = []
for f in audio_files:
    text = os.path.splitext(f)[0].replace("_", " ")  # Extract text from filename
    path = os.path.join(DATASET_PATH, f)
    data.append({"path": path, "text": text})

df = pd.DataFrame(data)

# Split into 80% train, 20% test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Save CSVs for use with datasets library
train_csv = "/kaggle/working/train.csv"
test_csv = "/kaggle/working/test.csv"
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# ================================
# STEP 2: LOAD DATASET
# ================================
dataset = load_dataset("csv", data_files={"train": train_csv, "test": test_csv})

# ================================
# STEP 3: LOAD MODEL & PROCESSOR
# ================================
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)
model.to(DEVICE)

# ================================
# STEP 4: PREPROCESSING FUNCTION
# ================================
def preprocess_function(batch):
    speech_array, _ = librosa.load(batch["path"], sr=SAMPLE_RATE)
    batch["input_values"] = processor(speech_array, sampling_rate=SAMPLE_RATE).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Map datasets (use num_proc=1 to avoid hanging)
dataset["train"] = dataset["train"].map(preprocess_function, remove_columns=["path", "text"], num_proc=1)
dataset["test"] = dataset["test"].map(preprocess_function, remove_columns=["path", "text"], num_proc=1)

# ================================
# STEP 5: METRICS
# ================================
wer_metric = load_metric("wer")
cer_metric = load_metric("cer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    total_tokens = 0
    correct_tokens = 0
    for p, l in zip(pred_ids, label_ids):
        mask = l != processor.tokenizer.pad_token_id
        total_tokens += mask.sum()
        correct_tokens += ((p == l) & mask).sum()
    token_acc = correct_tokens / total_tokens

    return {"wer": wer, "cer": cer, "token_accuracy": token_acc}

# ================================
# STEP 6: TRAINING ARGUMENTS
# ================================
training_args = TrainingArguments(
    output_dir="./svla-finetuned",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    save_total_limit=2,
    fp16=FP16,
    report_to="none"
)

# ================================
# STEP 7: TRAINER
# ================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

# ================================
# STEP 8: TRAIN & EVALUATE
# ================================
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)
