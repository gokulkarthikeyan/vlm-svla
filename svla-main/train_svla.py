%%writefile train_svla.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import json
import os

# =============================
# Custom dataset
# =============================
class SvlADataset(Dataset):
    def __init__(self, path, tokenizer):
        with open(path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = f"<image>{item['text']} {' '.join(item['speech_tokens'])}"
        inputs = self.tokenizer(prompt, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        labels = self.tokenizer(item['response'], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": labels["input_ids"].squeeze()
        }

# =============================
# Load model and tokenizer
# =============================
model_path = "./weights/svla-sft-text-ins"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# =============================
# Prepare datasets
# =============================
train_dataset = SvlADataset("dataset/train.json", tokenizer)
val_dataset = SvlADataset("dataset/val.json", tokenizer)

# =============================
# Training Arguments
# =============================
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=torch.cuda.is_available(),
)

# =============================
# Trainer
# =============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# =============================
# Start Training
# =============================
trainer.train()

trainer.save_model("./trained_svla")
print("✅ Training completed and model saved in ./trained_svla")
