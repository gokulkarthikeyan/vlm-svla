import torch
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2ForCTC, 
    Wav2Vec2Processor, 
    TrainingArguments, 
    Trainer
)
import librosa
import numpy as np

# ================================
# CONFIGURATION
# ================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP16 = True if DEVICE == "cuda" else False
SAMPLE_RATE = 16000
BATCH_SIZE = 2          # Small batch for CPU/GPU
EPOCHS = 3
LEARNING_RATE = 1e-4

# ================================
# LOAD LIBRISPEECH DATASET (subset for Kaggle)
# ================================
dataset = load_dataset("librispeech_asr", "clean", split="train.100")
dataset_test = load_dataset("librispeech_asr", "clean", split="test.clean")

# Rename columns to match preprocessing
dataset = dataset.rename_column("file", "path")
dataset = dataset.rename_column("text", "text")
dataset_test = dataset_test.rename_column("file", "path")
dataset_test = dataset_test.rename_column("text", "text")

dataset_dict = {"train": dataset, "test": dataset_test}

# ================================
# LOAD MODEL & PROCESSOR
# ================================
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base-960h",
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id
)
model.to(DEVICE)

# ================================
# PREPROCESS FUNCTION
# ================================
def preprocess_function(batch):
    speech_array, _ = librosa.load(batch["path"], sr=SAMPLE_RATE)
    batch["input_values"] = processor(speech_array, sampling_rate=SAMPLE_RATE).input_values[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch

# Map dataset (num_proc=1 to avoid hanging on Kaggle)
dataset_dict["train"] = dataset_dict["train"].map(
    preprocess_function, remove_columns=["path", "text"], num_proc=1
)
dataset_dict["test"] = dataset_dict["test"].map(
    preprocess_function, remove_columns=["path", "text"], num_proc=1
)

# ================================
# METRICS
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
# TRAINING ARGUMENTS
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
# TRAINER
# ================================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["test"],
    tokenizer=processor.feature_extractor,
    compute_metrics=compute_metrics
)

# ================================
# TRAIN & EVALUATE
# ================================
trainer.train()
results = trainer.evaluate()
print("Evaluation Results:", results)
