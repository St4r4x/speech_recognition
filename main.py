import argparse
import json
import os

import torch
from pydub import AudioSegment
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def init_model():
    """Initialize the Whisper model and processor"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,
    )


def convert_to_wav(input_path):
    """Convert any audio file to WAV format"""
    try:
        file_extension = os.path.splitext(input_path)[1].lower()
        if file_extension == '.wav':
            return input_path

        audio = AudioSegment.from_file(input_path)
        wav_path = os.path.splitext(input_path)[0] + '.wav'
        audio.export(wav_path, format='wav')
        print(f"Converted {input_path} to WAV format")
        return wav_path

    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return None


def transcribe_audio(pipe, audio_path):
    """Transcribe audio file using the pipeline"""
    try:
        result = pipe(audio_path)
        return result
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None


def format_dialogue_from_timestamps(result):
    """Format the transcription as a dialogue using timestamps"""
    dialogue_lines = []

    if "chunks" in result and isinstance(result["chunks"], list):
        for chunk in result["chunks"]:
            if "timestamp" in chunk:
                start_time = chunk["timestamp"][0]
                end_time = chunk["timestamp"][1]
                # Convertir les timestamps en format MM:SS
                start_fmt = f"{int(start_time//60):02d}:{int(start_time % 60):02d}"
                end_fmt = f"{int(end_time//60):02d}:{int(end_time % 60):02d}"
                text = chunk["text"].strip()
                dialogue_lines.append(f"[{start_fmt} - {end_fmt}] {text}")

    return "\n\n".join(dialogue_lines)


def save_results(result, output_path):
    """Save transcription results to a file"""
    try:
        # Créer le dossier de sortie s'il n'existe pas
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Sauvegarder en format texte brut
        txt_path = output_path + ".txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])

        # Sauvegarder en format dialogue
        dialogue_path = output_path + "_dialogue.txt"
        with open(dialogue_path, "w", encoding="utf-8") as f:
            f.write("=== Transcription avec timestamps ===\n\n")
            f.write(format_dialogue_from_timestamps(result))

        # Sauvegarder en format JSON
        json_path = output_path + ".json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Results saved to:")
        print(f"- Raw text: {txt_path}")
        print(f"- Dialogue format: {dialogue_path}")
        print(f"- JSON with timestamps: {json_path}")
        return True

    except Exception as e:
        print(f"Error saving results: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Audio transcription using Whisper")
    parser.add_argument("audio_path", help="Path to the audio file")
    parser.add_argument(
        "--output", "-o", help="Output path for results", default="results")
    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file {args.audio_path} not found")
        return

    # Initialize model
    print("Initializing model...")
    pipe = init_model()

    # Convert audio to WAV if needed
    wav_path = convert_to_wav(args.audio_path)
    if not wav_path:
        return

    # Transcribe audio
    print("Transcribing audio...")
    result = transcribe_audio(pipe, wav_path)
    if not result:
        return

    # Save results
    output_base = os.path.join(args.output, os.path.splitext(
        os.path.basename(args.audio_path))[0])
    save_results(result, output_base)


if __name__ == "__main__":
    main()
