import os
import torch
from PIL import Image
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN

# -------------------- device --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- find image --------------------
img_path = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f.lower() == "dog.jpeg":   # adjust name if needed
            img_path = os.path.join(root, f)
            break
if img_path is None:
    raise FileNotFoundError("dog.jpeg not found in /kaggle/input")

print("Using image:", img_path)
image = Image.open(img_path).convert("RGB")

# resize if too large
max_dim = 896
w, h = image.size
if max(w, h) > max_dim:
    scale = max_dim / max(w, h)
    image = image.resize((int(w * scale), int(h * scale)))

# -------------------- load model --------------------
MODEL_PATH = "./weights/svla-sft-text-ins"  # make sure weights are unzipped here
model = LlavaQwen2ForCausalLM.from_pretrained(
    MODEL_PATH, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True
)
vision_tower = model.get_vision_tower()
vision_tower.load_model(device_map="auto")
image_processor = vision_tower.image_processor
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# -------------------- prepare prompt --------------------
prompt_text = "Describe this image."
formatted_prompt = (
    f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    f"<|im_start|>user\n{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n"
    f"{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
)

# -------------------- generate --------------------
input_ids = tokenizer([formatted_prompt], return_tensors="pt", add_special_tokens=False).to(device)
img_tensor = image_processor(image, return_tensors="pt")["pixel_values"].to(device)
outputs = model.generate(
    inputs=input_ids["input_ids"],
    images=img_tensor,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

print("\nOUTPUT:\n", response)
