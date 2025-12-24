import cv2
import pyaudio
import subprocess
import threading
import json
import base64
import time
import os
import sys
import wave
import numpy as np
from scipy import signal
from gtts import gTTS
from io import BytesIO
import whisper

# Configuration
MODEL_PATH = "gemma-3n-E2B"
# MODEL_PATH = "gemma-3n-E4B"
LIT_BINARY = "./lit.macos_arm64"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
SAMPLE_RATE = 48000  # High quality audio (MacBook Air default)
CHANNELS = 1
CHUNK_SIZE = 4096  # Larger chunk for better quality
AUDIO_DIR = os.path.abspath("audio_inputs")
IMAGE_DIR = os.path.abspath("image_inputs")

# Create directories for inputs
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

class WhisperSTT:
    """Speech-to-text using Whisper"""
    def __init__(self, model_size="tiny"):
        # Use CPU for Whisper (MPS has compatibility issues with sparse tensors)
        # tiny model is fast enough on CPU for real-time use
        print(f"[INFO] Loading Whisper {model_size} model on CPU...")
        self.model = whisper.load_model(model_size, device="cpu")
        print(f"[INFO] Whisper model loaded (using CPU)")

    def transcribe(self, audio_path):
        """Transcribe audio file to text"""
        print(f"[INFO] Transcribing audio...")
        result = self.model.transcribe(audio_path, language="en", fp16=False)
        text = result["text"].strip()
        print(f"[INFO] Transcription: '{text}'")
        return text

class TTSPlayer:
    def speak(self, text):
        if not text.strip():
            return
        try:
            # Use macOS native 'say' for faster response
            subprocess.run(["say", "-r", "200", text])
        except Exception as e:
            print(f"TTS Error: {e}")

class AudioRecorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()

        # Find the best input device (prefer built-in microphone)
        input_device_index = None
        print("[INFO] Available audio devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} - {info['maxInputChannels']} channels, {info['defaultSampleRate']} Hz")
                if 'built-in' in info['name'].lower() or 'macbook' in info['name'].lower():
                    input_device_index = i
                    print(f"  --> Selected device: {info['name']}")

        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=CHUNK_SIZE
        )
        self.frames = []
        self.recording = False
        self.lock = threading.Lock()

        if input_device_index is not None:
            device_info = self.p.get_device_info_by_index(input_device_index)
            print(f"[INFO] Recording with: {device_info['name']} at {SAMPLE_RATE}Hz")

    def start(self):
        self.recording = True
        self.frames = []
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        while self.recording:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                with self.lock:
                    self.frames.append(data)
            except:
                break

    def stop_and_save(self, filename):
        self.recording = False
        if hasattr(self, 'thread'):
            self.thread.join()

        with self.lock:
            if not self.frames:
                return False

            # Convert audio data to numpy array
            audio_data = np.frombuffer(b''.join(self.frames), dtype=np.int16)

            # Resample to 16kHz for the model (speech recognition standard)
            target_rate = 16000
            if SAMPLE_RATE != target_rate:
                # Calculate resampling ratio
                num_samples = int(len(audio_data) * target_rate / SAMPLE_RATE)
                audio_data = signal.resample(audio_data, num_samples).astype(np.int16)
                print(f"[INFO] Resampled audio from {SAMPLE_RATE}Hz to {target_rate}Hz")

            # Save resampled audio
            wf = wave.open(filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(target_rate)  # Save at 16kHz
            wf.writeframes(audio_data.tobytes())
            wf.close()
            self.frames = []
            return True

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

class ModelInterface:
    def __init__(self, model_path, binary_path, on_response):
        self.model_path = model_path
        self.binary_path = binary_path
        # Use GPU backend now that we're doing STT separately
        self.cmd = [
            binary_path,
            "run",
            model_path,
            "--backend", "gpu"
        ]
        self.on_response = on_response
        self.running = True
        self._start_process()

    def _start_process(self):
        self.process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1 # Line buffered
        )
        self.thread = threading.Thread(target=self._listen)
        self.thread.start()

    def _listen(self):
        while self.running and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    self.on_response(line)
                else:
                    if self.process.poll() is not None:
                        break
            except Exception as e:
                print(f"[ERROR] Listen thread error: {e}")
                break

        # Check for errors when process ends
        if self.process.poll() is not None:
            stderr_output = self.process.stderr.read()
            if stderr_output:
                print(f"[ERROR] Model process stderr:\n{stderr_output}")
        
    def send_text_and_image(self, text, image_path=None):
        # The CLI expects JSON format with role and content array
        # Build the content array with image and text

        content = []

        if image_path:
            content.append({"type": "image", "path": image_path})

        # Add the text (already includes transcription)
        instruction = "You are having a video call. Based on what you see and what the user said, respond naturally and conversationally in 1-2 sentences. Do not use emojis. "

        if text:
            full_text = instruction + f"User said: \"{text}\""
        else:
            full_text = instruction + "Respond to what you see."

        content.append({"type": "text", "text": full_text})

        # Create JSON message
        message = {
            "role": "user",
            "content": content
        }

        msg = json.dumps(message)

        try:
            print(f"[DEBUG] Sending to model: {msg[:200]}...") # Debug log
            self.process.stdin.write(msg + "\n")
            self.process.stdin.flush()
        except BrokenPipeError:
            print("Model process disconnected")
            self.running = False


class VideoDisplay:
    def __init__(self, model_interface, whisper_stt):
        self.cap = cv2.VideoCapture(0)
        self.model = model_interface
        self.whisper = whisper_stt
        self.audio = AudioRecorder()
        self.running = True
        self.recording_audio = False
        self.turn_counter = 0

        print(" Controls:")
        print(" [r]     - Toggle recording (press to start, press again to send)")
        print(" [q]     - Quit")

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # visual indicator for recording
            if self.recording_audio:
                cv2.circle(display_frame, (30, 30), 20, (0, 0, 255), -1)
            
            cv2.imshow('Gemma Live Call', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '): # Space bar pressed
                if not self.recording_audio:
                    print("Start recording...")
                    self.recording_audio = True
                    self.audio.start()
            
            # Detect key release is hard in OpenCV waitKey loop without events, 
            # so we use a toggle or simple check. 
            # To implement "Hold to talk", we need a different approach or assume toggle.
            # Let's use TOGGLE for simplicity: Press Space to start, Press Space again to stop?
            # Or use a fixed duration?
            # The prompt implied "live streaming". 
            # Let's do: Press 'r' to toggle recording.
            
            if key == ord('r'):
                if not self.recording_audio:
                    print("Start Recording Audio...")
                    self.recording_audio = True
                    self.audio.start()
                else:
                    print("Stop Recording & Send...")
                    self.recording_audio = False

                    # Use unique filenames for each turn
                    self.turn_counter += 1
                    audio_file = os.path.join(AUDIO_DIR, f"audio_{self.turn_counter}.wav")
                    image_file = os.path.join(IMAGE_DIR, f"image_{self.turn_counter}.jpg")

                    if self.audio.stop_and_save(audio_file):
                        # Also save the current frame
                        cv2.imwrite(image_file, frame)

                        # Debug info
                        try:
                            img_size = os.path.getsize(image_file) / 1024.0 # KB
                            with wave.open(audio_file, 'rb') as wf:
                                duration = wf.getnframes() / float(wf.getframerate())

                            print(f"[DEBUG] Turn {self.turn_counter} - Sending Media:")
                            print(f" - Image: {image_file} ({FRAME_WIDTH}x{FRAME_HEIGHT}, {img_size:.2f} KB)")
                            print(f" - Audio: {audio_file} ({duration:.2f}s)")
                        except Exception as e:
                            print(f"[DEBUG] Error reading media stats: {e}")

                        # Transcribe audio to text using Whisper
                        transcribed_text = self.whisper.transcribe(audio_file)

                        # Send text + image to model (keeping conversation history)
                        self.model.send_text_and_image(
                            text=transcribed_text,
                            image_path=image_file
                        )

            if key == ord('q'):
                self.running = False
        
        self.cap.release()
        self.audio.close()
        cv2.destroyAllWindows()

def main():
    # Initialize Whisper for speech-to-text
    # Using "tiny" model for fastest CPU performance (~1 second transcription)
    # Can use "base" for better accuracy but slower (~3-4 seconds)
    whisper_stt = WhisperSTT(model_size="tiny")

    # Initialize TTS
    tts = TTSPlayer()

    def handle_response(text):
        text = text.strip()
        print(f"Model: {text}")
        # Improve filter to ignore system logs
        if not text: return
        if text.startswith("Model"): return # Initiation line
        if text.startswith("Stream error"):
            print("!! Streaming Error - Request too large?")
            return

        tts.speak(text)

    model = ModelInterface(MODEL_PATH, LIT_BINARY, handle_response)
    video = VideoDisplay(model, whisper_stt)

    try:
        video.run()
    except KeyboardInterrupt:
        pass
    finally:
        model.running = False

if __name__ == "__main__":
    main()
