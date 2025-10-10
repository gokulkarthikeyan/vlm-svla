# =====================================================
# IMPORTS
# =====================================================
import os
import numpy as np
import librosa
from datasets import Dataset, logging as hf_logging
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer
)
import evaluate

# =====================================================
# SUPPRESS DATASET WARNINGS FOR CLEAN OUTPUT
# =====================================================
hf_logging.set_verbosity_error()

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
# LOAD AUDIO FILES
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
# FEATURE EXTRACTION FUNCTION (batched + multi-core)
# =====================================================
def preprocess(batch):
    input_values = []
    labels = []

    for path, text in zip(batch["path"], batch["text"]):
        # Load audio
        speech_array, _ = librosa.load(path, sr=SAMPLE_RATE)
        
        # Convert audio to input_values for Wav2Vec2
        input_values.append(processor(speech_array, sampling_rate=SAMPLE_RATE).input_values[0])
        
        # Convert text to labels
        with processor.as_target_processor():
            labels.append(processor(text).input_ids)

    batch["input_values"] = input_values
    batch["labels"] = labels
    return batch

# =====================================================
# APPLY PREPROCESSING USING MULTI-CORE + BATCHING
# =====================================================
dataset = dataset.map(
    preprocess,
    remove_columns=["path"],  # remove file paths
    batched=True,             # process multiple examples at once
    batch_size=4,             # adjust based on Kaggle memory
    num_proc=4                # parallel processing
)
print("✅ Preprocessing complete!")

# =====================================================
# METRIC: WORD ERROR RATE (WER)
# =====================================================
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred_str = processor.batch_decode(pred_ids)
    
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
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
