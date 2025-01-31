# Speech Recognition with Pyannote.audio

Ce projet utilise pyannote.audio pour effectuer de la diarisation de locuteurs sur des fichiers audio.

## Installation

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

2. Installer FFmpeg (nécessaire pour la conversion audio) :

```bash
# Sur Ubuntu/Debian
sudo apt-get install ffmpeg

# Sur MacOS avec Homebrew
brew install ffmpeg

# Sur Windows
# Télécharger et installer depuis https://ffmpeg.org/download.html
```

3. Configurer le token HuggingFace :
   - Créez un fichier `.env` à la racine du projet
   - Ajoutez votre token HuggingFace sous la forme :
   ```
   HUGGING_FACE_TOKEN=votre_token_ici
   ```

## Utilisation

```bash
python speech_recognition.py chemin/vers/audio.[wav|mp3|ogg|etc]
```

Le script supporte différents formats audio en entrée (WAV, MP3, OGG, etc.) et les convertira automatiquement en WAV si nécessaire.

Le résultat sera sauvegardé dans un fichier RTTM avec le même nom que le fichier audio.
