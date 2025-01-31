# Speech Recognition with Whisper

Ce projet utilise Whisper (OpenAI) pour effectuer de la transcription automatique de fichiers audio avec horodatage.

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

## Utilisation

```bash
python main.py chemin/vers/audio.[wav|mp3|ogg|etc] [--output dossier/resultats]
```

Options :

- `--output`, `-o` : Spécifie le dossier de sortie pour les résultats (défaut: "results")

### Format des sorties

Le script génère trois fichiers de sortie :

1. `xxx.txt` : Transcription brute du texte
2. `xxx_dialogue.txt` : Transcription formatée avec horodatage [MM:SS]
3. `xxx.json` : Données complètes incluant :
   - Texte complet
   - Timestamps pour chaque segment
   - Informations détaillées sur la transcription

### Formats supportés

Le script accepte de nombreux formats audio en entrée (WAV, MP3, OGG, etc.) et les convertit automatiquement en WAV si nécessaire.

### Exemple de sortie dialogue

```
=== Transcription avec timestamps ===

[00:00 - 00:15] Début de la transcription...
[00:15 - 00:30] Suite de la conversation...
```

## Notes

- Le modèle utilisé est "whisper-large-v3-turbo" d'OpenAI
- La transcription s'effectue sur GPU si disponible, sinon sur CPU
- Les fichiers temporaires WAV sont automatiquement gérés
