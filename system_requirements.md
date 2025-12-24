# System Requirements

## Homebrew Packages

Run the following to install system-level dependencies:

```bash
brew install portaudio ffmpeg
```

- **portaudio**: Required for `pyaudio` (audio recording)
- **ffmpeg**: Required for `openai-whisper` (audio file processing)

## Python Packages

Install Python dependencies from `requirements.txt`:

```bash
conda activate cv  # or your conda environment
pip install -r requirements.txt
```

This will install:
- **pyaudio**: Audio recording from microphone
- **opencv-python**: Video capture and image processing
- **gTTS**: Text-to-speech (fallback option)
- **numpy**: Numerical operations for audio processing
- **scipy**: Audio resampling
- **openai-whisper**: Speech-to-text transcription

## Camera & Microphone Permissions

This application requires access to your Mac's Camera and Microphone.

- **Grant Permission**: When running the script for the first time, macOS should prompt you to allow Terminal/IDE to access the Camera and Microphone.
- **Manual Enable**: If denied, go to **System Settings > Privacy & Security > Camera / Microphone** and enable access for your terminal application (e.g., iTerm, VSCode, Terminal).

## LiteRT-LM Model & Binary

- **Model**: `gemma-3n-E2B-it-int4.litertlm` (must be present in the project root)
- **Binary**: `lit.macos_arm64` (must be executable: `chmod +x lit.macos_arm64`)
