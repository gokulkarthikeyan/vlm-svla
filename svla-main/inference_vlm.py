import torch
import torchaudio
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer,
)
import evaluate
import numpy as np
import argparse
import os

# ====================================================
# ARGUMENT PARSER
# ====================================================
parser = argparse.ArgumentParser(description="ASR Training & Evaluation on LibriSpeech")
parser.add_argument("--mode", type=str, default="train_asr", choices=["train_asr", "eval_asr", "infer"],
                    help="Mode: train_asr | eval_asr | infer")
parser.add_argument("--audio_file", type=str, help="Path to audio file for inference")
args = parser.parse_args()

# ====================================================
# CONFIGURATION
# ====================================================
MODEL_NAME = "facebook/wav2vec2-base-960h"
OUTPUT_DIR = "./asr_librispeech_model"
CACHE_DIR = "./data"

# ====================================================
# LOAD DATASET
# ====================================================
def load_librispeech_datasets():
    print("🔹 Loading LibriSpeech dataset...")
    dataset = load_dataset("librispeech_asr", "clean", cache_dir=CACHE_DIR)
    train_dataset = dataset["train.clean.100"]
    test_dataset = dataset["test.clean"]
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    return train_dataset, test_dataset

# ====================================================
# PREPARE MODEL & PROCESSOR
# ====================================================
def load_model_and_processor():
    print("🔹 Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForCTC.from_pretrained(MODEL_NAME)
    return model, processor

# ====================================================
# PREPROCESS DATA
# ====================================================
def prepare_dataset(batch, processor):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

# ====================================================
# METRICS (WER + CER)
# ====================================================
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
    return {"wer": wer, "cer": cer}

# ====================================================
# TRAINING FUNCTION
# ====================================================
def train_asr_model():
    train_dataset, test_dataset = load_librispeech_datasets()
    model, processor = load_model_and_processor()

    print("🔹 Preprocessing data...")
    train_dataset = train_dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=test_dataset.column_names)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_strategy="epoch",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    data_collator = lambda data: {
        "input_values": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["input_values"]) for f in data], batch_first=True, padding_value=0
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(f["labels"]) for f in data], batch_first=True, padding_value=-100
        ),
    }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("🔹 Starting training...")
    trainer.train()

    print("🔹 Evaluating final model...")
    metrics = trainer.evaluate()
    print(f"✅ Final Evaluation Results: WER = {metrics['eval_wer']:.3f}, CER = {metrics['eval_cer']:.3f}")

    print("💾 Saving model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")

# ====================================================
# EVALUATION FUNCTION
# ====================================================
def evaluate_asr_model():
    _, test_dataset = load_librispeech_datasets()
    model, processor = load_model_and_processor()
    test_dataset = test_dataset.map(lambda x: prepare_dataset(x, processor), remove_columns=test_dataset.column_names)

    trainer = Trainer(model=model, tokenizer=processor.feature_extractor, compute_metrics=compute_metrics)
    print("🔹 Evaluating model on test set...")
    metrics = trainer.evaluate(eval_dataset=test_dataset)
    print(f"✅ Evaluation complete: WER = {metrics['eval_wer']:.3f}, CER = {metrics['eval_cer']:.3f}")

# ====================================================
# INFERENCE FUNCTION
# ====================================================
def infer_single_audio(audio_file):
    print(f"🔹 Running inference on: {audio_file}")
    model, processor = load_model_and_processor()
    model.eval()

    speech_array, sampling_rate = torchaudio.load(audio_file)
    inputs = processor(speech_array.squeeze(), sampling_rate=sampling_rate, return_tensors="pt")

    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(pred_ids)[0]
    print(f"✅ Transcription: {transcription}")

# ====================================================
# MAIN
# ====================================================
if args.mode == "train_asr":
    train_asr_model()
elif args.mode == "eval_asr":
    evaluate_asr_model()
elif args.mode == "infer":
    if not args.audio_file:
        print("❌ Please provide an --audio_file for inference.")
    else:
        infer_single_audio(args.audio_file)
