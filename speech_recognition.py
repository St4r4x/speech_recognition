import argparse
import os

import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pydub import AudioSegment

# Charger les variables d'environnement
load_dotenv()


def convert_to_wav(input_path):
    """Convert any audio file to WAV format"""
    try:
        # Get the file extension
        file_extension = os.path.splitext(input_path)[1].lower()

        # If already WAV, return original path
        if file_extension == '.wav':
            return input_path

        # Convert to WAV
        audio = AudioSegment.from_file(input_path)
        wav_path = os.path.splitext(input_path)[0] + '.wav'
        audio.export(wav_path, format='wav')
        print(f"Converted {input_path} to WAV format")
        return wav_path

    except Exception as e:
        print(f"Error converting audio: {str(e)}")
        return None


def process_audio(audio_path):
    try:
        # Convert to WAV first
        wav_path = convert_to_wav(audio_path)
        if not wav_path:
            return

        # Récupérer le token depuis .env
        token = os.getenv('HUGGING_FACE_TOKEN')
        if not token:
            raise ValueError(
                "HUGGING_FACE_TOKEN non trouvé dans le fichier .env")

        # Instantiate the pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token
        )
        print(torch.cuda.is_available())

        # Use GPU if requested and available
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))

        # Run the pipeline on the WAV file
        diarization = pipeline(wav_path)

        # Create output filename
        output_path = os.path.splitext(audio_path)[0] + ".rttm"

        # Save results to RTTM file
        with open(output_path, "w") as rttm:
            diarization.write_rttm(rttm)

        print(f"Diarization completed successfully. Results saved to {
              output_path}")

        # Clean up temporary WAV file if it was converted
        # if wav_path != audio_path:
        #     os.remove(wav_path)
        #     print(f"Cleaned up temporary WAV file")

    except Exception as e:
        print(f"An error occurred: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Speaker diarization using pyannote.audio")
    parser.add_argument("audio_path", help="Path to the audio file")

    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file {args.audio_path} not found")
        return

    process_audio(args.audio_path)


if __name__ == "__main__":
    main()
