import argparse
import torch
from transformers import AutoTokenizer
from llava.model import LlavaQwen2ForCausalLM
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from PIL import Image

# -------------------------------
# 1. Argument parser
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SVLA Inference Script")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, default=None, help="Optional question for VQA")
    parser.add_argument("--model-path", type=str, default="./weights/svla-sft-text-ins", help="Path to model weights")
    return parser.parse_args()

# -------------------------------
# 2. Run inference
# -------------------------------
def run_inference(image_path, model_path, question=None):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = LlavaQwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    image_processor = model.get_vision_tower().image_processor

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)

    def ask_question(q):
        conv = conv_templates["qwen2_vl"].copy()
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + q
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX=tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        ).unsqueeze(0).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.to(model.device, dtype=torch.float16),
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
            )

        outputs = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        return outputs

    # If question is given from CLI, run once
    if question:
        print(f"\nQ: {question}")
        print(f"A: {ask_question(question)}\n")
    else:
        # Otherwise enter interactive mode
        print("\n✅ VQA model is ready! Type your questions (type 'exit' to quit)\n")
        while True:
            q = input("Q: ")
            if q.strip().lower() in ["exit", "quit"]:
                print("Exiting VQA session.")
                break
            print(f"A: {ask_question(q)}\n")


# -------------------------------
# 3. Main
# -------------------------------
if __name__ == "__main__":
    args = parse_args()
    run_inference(args.image, args.model_path, args.question)
