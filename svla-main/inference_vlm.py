# =====================================================
# IMPORTS
# =====================================================
import os
import numpy as np
from datasets import Dataset, Audio
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import evaluate

# =====================================================
# CONFIGURATION
# =====================================================
DATA_DIR = "/kaggle/input/librispeeech/LibriSpeech/train-clean-100"
MODEL_NAME = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
MAX_FILES = 200  # small subset for Kaggle

# =====================================================
# HELPER FUNCTION TO LOAD AUDIO + TRANSCRIPTS
# =====================================================
def load_librispeech_subset(path, max_files=None):
    audio_paths = []
    texts = []

    for root, _, files in os.walk(path):
        transcript_files = [f for f in files if f.endswith(".trans.txt")]
        for tf in transcript_files:
            with open(os.path.join(root, tf), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        audio_file = os.path.join(root, parts[0] + ".flac")
                        if os.path.exists(audio_file):
                            audio_paths.append(audio_file)
                            texts.append(parts[1])
                    if max_files and len(audio_paths) >= max_files:
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
dataset = dataset.cast_column("path", Audio(sampling_rate=SAMPLE_RATE))
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
    # batch["path"]["array"] is from datasets Audio type
    batch["input_values"] = processor(batch["path"]["array"], sampling_rate=SAMPLE_RATE).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

dataset = dataset.map(preprocess, remove_columns=["path"], num_proc=2)

# =====================================================
# METRIC: WORD ERROR RATE (WER)
# =====================================================
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id  # handle -100
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# =====================================================
# TRAINING ARGUMENTS
# =====================================================
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
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
    tokenizer=processor,
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
