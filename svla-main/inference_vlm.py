import torch
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image

# -------------------------------
# 1. Load model & tokenizer
# -------------------------------
disable_torch_init()
model_path = "./weights/svla-sft-text-ins"   # adjust if needed
model_name = get_model_name_from_path(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = LlavaQwen2ForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
image_processor = model.get_vision_tower().image_processor

# -------------------------------
# 2. Load image
# -------------------------------
image_file = "./examples/dogs.jpg"   # change to your test image
image = Image.open(image_file).convert("RGB")
image_tensor = process_images([image], image_processor, model.config)

# -------------------------------
# 3. Interactive QA loop
# -------------------------------
print("\n✅ VQA model is ready! Type your questions (type 'exit' to quit)\n")

while True:
    question = input("Q: ")
    if question.strip().lower() in ["exit", "quit"]:
        print("Exiting VQA session.")
        break

    # Conversation template
    conv = conv_templates["qwen2_vl"].copy()
    qs = question
    qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize & forward
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX=tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(model.device, dtype=torch.float16),
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )

    # Decode
    outputs = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    print(f"A: {outputs}\n")
