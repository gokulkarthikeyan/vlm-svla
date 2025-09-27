import argparse
import torch
from PIL import Image
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from inference.tokens_to_audio import decode_speech
import random
import librosa

MODEL_PATH = "./weights/svla-sft-text-ins"
SPEECH_OUTPUT_PATH = "speech_question.wav"
ANSWER_SPEECH_PATH = "speech_answer.wav"
SPEED = 1.0

def load_model(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(
        model_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True
    )
    model.get_vision_tower().load_model(device_map="cuda:0")
    return model

def preprocess_image(model, image):
    # Use SVLA/LLaVA helper to preprocess image
    image_tensor = model.preprocess_images([image]).to("cuda:0")
    return image_tensor[0]  # Single image

def generate_answer(model, image_tensor, question, max_tokens=1024):
    # Prepare prompt
    system_prompt = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
    formatted_prompt = f"{system_prompt}\n<|im_start|>user\n<|im_start|>image\n{question}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize and generate
    tokenizer = model.get_tokenizer()
    input_ids = tokenizer([formatted_prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda:0")

    outputs = model.generate(
        inputs=input_ids,
        images=image_tensor.unsqueeze(0),
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=1.0,
        repetition_penalty=1.3,
        do_sample=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--question", type=str, required=True, help="Question for the VQA model")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to SVLA model weights")
    args = parser.parse_args()

    # Load image
    image = Image.open(args.image).convert("RGB")

    # Load model
    print("Loading model...")
    model = load_model(args.model_path)

    # Preprocess image
    image_tensor = preprocess_image(model, image)

    # Generate answer
    print(f"Question: {args.question}")
    answer = generate_answer(model, image_tensor, args.question)
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
