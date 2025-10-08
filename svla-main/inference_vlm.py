import torch
from PIL import Image, UnidentifiedImageError
import requests, os, random
from transformers import AutoTokenizer, Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from transformers import CLIPImageProcessor, SiglipImageProcessor
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from llava.model import LlavaQwen2ForCausalLM
from melo.api import TTS
from inference.audio_encoder import audio_encoder
from inference.tokens_to_audio import decode_speech
from llava.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from datasets import load_dataset, load_metric
import librosa

MODEL_PATH = "./weights/svla-sft-text-ins"
IMAGE_TOKEN_INDEX = -200
speech_output_path = "speech_question.wav"
speed = 1.0

# ------------------ Utility Functions ------------------
def get_enhanced_input(prompt_text=">>> ", history_file=".input_history", completer_words=None):
    completer = WordCompleter(completer_words) if completer_words else None
    try:
        return prompt(prompt_text, history=FileHistory(history_file),
                      auto_suggest=AutoSuggestFromHistory(), completer=completer)
    except (KeyboardInterrupt, EOFError):
        return None

def resize_image_if_necessary(image):
    original_width, original_height = image.size
    longest_dimension = 896
    if original_width <= longest_dimension and original_height <= longest_dimension:
        return image
    if original_width > original_height:
        new_width = longest_dimension
        new_height = int((longest_dimension / original_width) * original_height)
    else:
        new_height = longest_dimension
        new_width = int((longest_dimension / original_height) * original_width)
    return image.resize((new_width, new_height))

def load_model_and_tokenizer(model_path):
    model = LlavaQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, device_map='cuda', trust_remote_code=True)
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    image_processor = vision_tower.image_processor
    return model, tokenizer, image_processor

def load_image_from_url(url):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        image = Image.open(response.raw)
        image.load()
        return resize_image_if_necessary(image)
    except:
        return None

def load_image_from_path(path):
    try:
        return resize_image_if_necessary(Image.open(path))
    except:
        return None

def generate_text(model, tokenizer, image, prompt, max_new_tokens=1024, temperature=0.7, top_p=1.0, repetition_penalty=1.3):
    if image is not None:
        image = image.unsqueeze(0).float().to("cuda:0")
    input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda:0")
    outputs = model.generate(inputs=input_ids, images=image, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty,
                             do_sample=temperature > 0.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ------------------ Main Function ------------------
def main():
    print("Loading VLMA model and tokenizer...")
    model, tokenizer, image_processor = load_model_and_tokenizer(MODEL_PATH)
    text_to_audio_model = TTS(language='EN', device="cuda:1")
    speaker_ids = text_to_audio_model.hps.data.spk2id

    # Load ASR model
    asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to("cuda")
    asr_model.eval()

    # ------------------ Mode Selection ------------------
    print("Kaggle mode: evaluating LibriSpeech automatically...")
    mode = "asr_eval"  # directly run ASR evaluation

    if mode.lower() == 'asr_eval':
        local_dataset_path = "/kaggle/working/LibriSpeech"
        train_dataset = load_dataset("librispeech_asr", "clean", split="train.100", cache_dir=local_dataset_path)
        test_dataset = load_dataset("librispeech_asr", "clean", split="test.clean", cache_dir=local_dataset_path)

        wer_metric = load_metric("wer")
        cer_metric = load_metric("cer")

        def batch_transcribe(batch):
            audios = [x["array"] for x in batch["audio"]]
            input_values = asr_tokenizer(audios, return_tensors="pt", padding=True, sampling_rate=16000).input_values.to("cuda")
            with torch.no_grad():
                logits = asr_model(input_values).logits
            pred_ids = torch.argmax(logits, dim=-1)
            batch["pred_text"] = asr_tokenizer.batch_decode(pred_ids)
            return batch

        print("Transcribing train-clean-100...")
        train_dataset = train_dataset.map(batch_transcribe, batched=True, batch_size=8)
        print("Transcribing test-clean...")
        test_dataset = test_dataset.map(batch_transcribe, batched=True, batch_size=8)

        def compute_metrics(dataset):
            references = [x["text"] for x in dataset]
            predictions = [x["pred_text"] for x in dataset]
            wer_score = wer_metric.compute(predictions=predictions, references=references)
            cer_score = cer_metric.compute(predictions=predictions, references=references)
            return wer_score, cer_score

        train_wer, train_cer = compute_metrics(train_dataset)
        test_wer, test_cer = compute_metrics(test_dataset)

        print("=== LibriSpeech ASR Evaluation ===")
        print(f"Train-clean-100: WER={train_wer:.4f}, CER={train_cer:.4f}")
        print(f"Test-clean:       WER={test_wer:.4f}, CER={test_cer:.4f}")
        return

    # ------------------ Interactive Assistant ------------------
    print("Interactive assistant mode activated...")
    while True:
        input_source = get_enhanced_input(prompt_text="Enter URL/file path ('quit' to exit): ")
        if input_source is None or input_source.lower() == "quit":
            break

        image = None
        if input_source.startswith('http'):
            image = load_image_from_url(input_source)
        elif os.path.exists(input_source):
            image = load_image_from_path(input_source)

        if image is not None:
            image = image_processor(image, return_tensors='pt')["pixel_values"][0]

        while True:
            prompt_text = get_enhanced_input("Enter prompt ('audio' for speech, 'end' to switch, 'quit' to exit): ")
            if prompt_text is None or prompt_text.lower() == 'quit':
                return
            if prompt_text.lower() == 'end':
                break

            if prompt_text.lower() in ['audio', 'speech']:
                text_input = get_enhanced_input("Type your speech here: ")
                if not text_input:
                    continue
                speaker = random.choice(['EN-US','EN-BR','EN_INDIA','EN-AU','EN-Default'])
                text_to_audio_model.tts_to_file(text_input, speaker_ids[speaker], speech_output_path, speed=speed)
                prompt_text = audio_encoder(speech_output_path)

            system = "<|im_start|>system\nYou are a helpful speech-text-vision assistant.<|im_end|>"
            formatted_prompt = f"{system}\n<|im_start|>user\n{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN*256}{DEFAULT_IM_END_TOKEN}\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n" if image is not None else f"{system}\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            generated_text = generate_text(model, tokenizer, image, formatted_prompt)
            print("************************************************* OUTPUT *************************************************")
            print(generated_text)
            print("---------------------------------------------------------------------------------------------------")

if __name__ == "__main__":
    main()
