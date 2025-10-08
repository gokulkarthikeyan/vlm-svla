import torch
import os
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
from jiwer import wer, cer

# ======== Device Setup ========
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ======== ASR Evaluation Function ========
def evaluate_librispeech(asr_model, asr_tokenizer, dataset_path):
    total_wer = 0.0
    total_cer = 0.0
    count = 0
    print(f"Starting evaluation on dataset: {dataset_path}")

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(".flac"):
                audio_path = os.path.join(root, f)
                label_path = audio_path.replace(".flac", ".txt")
                if not os.path.exists(label_path):
                    continue
                with open(label_path, 'r') as lf:
                    true_text = lf.read().strip().lower()
                
                audio, sr = librosa.load(audio_path, sr=16000)
                input_values = asr_tokenizer(audio, return_tensors="pt", padding="longest").input_values.to(device)
                
                with torch.no_grad():
                    logits = asr_model(input_values).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = asr_tokenizer.decode(predicted_ids[0]).lower()
                
                total_wer += wer(true_text, transcription)
                total_cer += cer(true_text, transcription)
                count += 1

                if count % 10 == 0:
                    print(f"Processed {count} samples... Current WER: {total_wer/count:.4f}, CER: {total_cer/count:.4f}")

    if count > 0:
        print(f"\nFinal evaluation on {dataset_path}: WER = {total_wer/count:.4f}, CER = {total_cer/count:.4f}\n")
    else:
        print("No valid audio files found in dataset.")

# ======== Main Function ========
def main():
    print("Starting automatic LibriSpeech evaluation (hands-free)...\n")
    
    print("Loading pretrained ASR model...")
    asr_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h")
    asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(device)
    asr_model.eval()

    train_path = "/kaggle/working/LibriSpeech/train-clean-100"
    test_path = "/kaggle/working/LibriSpeech/test-clean"

    print("\nEvaluating train-clean-100...")
    evaluate_librispeech(asr_model, asr_tokenizer, train_path)

    print("\nEvaluating test-clean...")
    evaluate_librispeech(asr_model, asr_tokenizer, test_path)

    print("LibriSpeech evaluation completed!")

if __name__ == "__main__":
    main()
