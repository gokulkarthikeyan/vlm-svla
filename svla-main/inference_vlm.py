# ================================
# IMPORTS
# ================================
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import torch
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer
import evaluate
from tqdm import tqdm

# ================================
# CONFIGURATION
# ================================
DATASET_PATH = "/kaggle/input/librispeech/LibriSpeech/train-clean-100"
BOOKS_TXT = "/kaggle/input/librispeech/LibriSpeech/book.txt"  # check case sensitivity!
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True if DEVICE == "cuda" else False
SAMPLE_RATE = 16000
BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 1e-4

# ================================
# STEP 1: VERIFY FILE PATHS
# ================================
print(f"✅ Checking dataset and files...")
print(f"DATASET_PATH exists: {os.path.exists(DATASET_PATH)}")
print(f"BOOKS.TXT exists: {os.path.exists(BOOKS_TXT)}")

# Show a few .flac files to confirm structure
found_files = []
for root, dirs, files in os.walk(DATASET_PATH):
    for f in files:
        if f.endswith(".flac"):
            found_files.append(os.path.join(root, f))
    if found_files:
        break

if not found_files:
    raise ValueError(f"❌ No .flac files found under {DATASET_PATH}. Check dataset structure!")
else:
    print(f"✅ Found sample .flac files:")
    for f in found_files[:5]:
        print(" -", f)

# ================================
# STEP 2: LOAD AUDIO & TRANSCRIPTS
# ================================
data = []

book_map = {}
if os.path.exists(BOOKS_TXT):
    with open(BOOKS_TXT, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                book_map[parts[0]] = parts[1].lower()

# Recursively scan .flac files
for root, dirs, files in os.walk(DATASET_PATH):
    for f in files:
        if f.endswith(".flac"):
            path = os.path.join(root, f)
            audio_id = os.path.splitext(f)[0]
            text = book_map.get(audio_id, audio_id.replace("_", " ").lower())
            data.append({"path": path, "text": text})

if len(data) == 0:
    raise ValueError("❌ No audio-transcript pairs found! Check BOOKS.TXT and dataset path.")

print(f"✅ Total audio samples found: {len(data)}")
df = pd.DataFrame(data)
print(df.head())

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

# Save CSVs
train_csv = "/kaggle/working/train.csv"
test_csv = "/kaggle/working/test.csv"
train_df.to_csv(train_csv, index=False)
test_df.to_csv(test_csv, index=False)

# ================================
# STEP 3: LOAD DATASET
# ================================
dataset = load_dataset("csv", data_files={"train": train_csv, "test": test_csv})

# ================================
# STEP 4: LOAD MODEL & PROCESSOR
# ================================
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)
model.to(DEVICE)

# ================================
# STEP 5: PREPROCESS FUNCTION
# ================================
def preprocess_function(batch):
    try:
        speech_array, _ = librosa.load(batch["path"], sr=SAMPLE_RATE)
        batch["input_values"] = processor(speech_array, sampling_rate=SAMPLE_RATE).input_values[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    except Exception as e:
        print(f"⚠️ Skipping {batch['path']} due to error: {e}")
        batch["input_values"] = []
        batch["labels"] = []
    return batch

# Preprocess with progress bar
for split in ["train", "test"]:
    dataset[split] = dataset[split].map(
        preprocess_function,
        remove_columns=["path", "text"],
        num_proc=1,
        desc=f"Preprocessing {split}"
    )

# ================================
# STEP 6: METRICS
# ================================
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

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
    token_acc = (correct_tokens / total_tokens).item() if total_tokens > 0 else 0

    return {"wer": wer, "cer": cer, "token_accuracy": token_acc}

# ================================
# STEP 7: TRAINING ARGUMENTS
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
# STEP 8: TRAINER
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
# STEP 9: TRAIN & EVALUATE
# ================================
trainer.train()
results = trainer.evaluate()
print("✅ Evaluation Results:", results)
