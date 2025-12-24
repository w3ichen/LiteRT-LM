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
import webrtcvad
from collections import deque
from enum import Enum

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

# UI States
class UIState(Enum):
    IDLE = "idle"           # Waiting for speech
    RECORDING = "recording" # Detecting speech
    PROCESSING = "processing" # Transcribing/Getting response

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
    def __init__(self):
        self.speaking = False
        self.lock = threading.Lock()
        # Use Samantha - more natural sounding voice on macOS
        # Other good options: "Alex", "Ava" (female), "Zoe" (female)
        self.voice = "Samantha"

    def speak(self, text):
        if not text.strip():
            return
        try:
            with self.lock:
                self.speaking = True
            print(f"[TTS] Speaking: {text[:50]}...")
            # Use natural voice with moderate speed (180 WPM)
            subprocess.run(["say", "-v", self.voice, "-r", "180", text])
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            with self.lock:
                self.speaking = False

    def is_speaking(self):
        with self.lock:
            return self.speaking

class VADAudioRecorder:
    """Voice Activity Detection based audio recorder"""
    def __init__(self, on_speech_end):
        self.p = pyaudio.PyAudio()
        self.on_speech_end = on_speech_end  # Callback when speech ends

        # VAD setup - needs 16kHz for WebRTC VAD
        self.vad_sample_rate = 16000
        self.vad = webrtcvad.Vad(2)  # Aggressiveness 0-3 (2 = moderate)

        # Find the best input device
        input_device_index = None
        print("[INFO] Available audio devices:")
        for i in range(self.p.get_device_count()):
            info = self.p.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"  [{i}] {info['name']} - {info['maxInputChannels']} channels, {info['defaultSampleRate']} Hz")
                if 'built-in' in info['name'].lower() or 'macbook' in info['name'].lower():
                    input_device_index = i
                    print(f"  --> Selected device: {info['name']}")

        # Open stream at 16kHz for VAD
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=self.vad_sample_rate,
            input=True,
            input_device_index=input_device_index,
            frames_per_buffer=480  # 30ms at 16kHz
        )

        # Speech detection state
        self.is_speaking = False
        self.speech_frames = []
        self.silence_frames = 0
        self.silence_threshold = 20  # ~600ms of silence to end speech
        self.speech_threshold = 5    # ~150ms of speech to start
        self.voice_buffer = deque(maxlen=10)  # Rolling buffer

        # Continuous monitoring
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()

        if input_device_index is not None:
            device_info = self.p.get_device_info_by_index(input_device_index)
            print(f"[INFO] Recording with: {device_info['name']} at {self.vad_sample_rate}Hz")

    def start_monitoring(self):
        """Start continuous monitoring for voice activity"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_audio, daemon=True)
        self.monitor_thread.start()
        print("[INFO] Started continuous voice monitoring")

    def _monitor_audio(self):
        """Continuously monitor audio for speech"""
        while self.monitoring:
            try:
                # Read 30ms frames (480 samples at 16kHz)
                frame = self.stream.read(480, exception_on_overflow=False)

                # Check if frame contains speech
                is_speech = self.vad.is_speech(frame, self.vad_sample_rate)

                with self.lock:
                    self.voice_buffer.append(is_speech)

                    # Count recent speech frames
                    recent_speech = sum(self.voice_buffer)

                    if not self.is_speaking:
                        # Not currently speaking - check if speech starts
                        if recent_speech >= self.speech_threshold:
                            self.is_speaking = True
                            self.speech_frames = []
                            self.silence_frames = 0
                            print("[VAD] Speech started")
                    else:
                        # Currently speaking - collect frames
                        self.speech_frames.append(frame)

                        if is_speech:
                            self.silence_frames = 0
                        else:
                            self.silence_frames += 1

                        # Check if speech ended (enough silence)
                        if self.silence_frames >= self.silence_threshold:
                            self._handle_speech_end()

            except Exception as e:
                print(f"[ERROR] VAD monitoring error: {e}")
                break

    def _handle_speech_end(self):
        """Called when speech ends - save and trigger callback"""
        if len(self.speech_frames) < 10:  # Too short, ignore
            self.is_speaking = False
            self.speech_frames = []
            return

        print(f"[VAD] Speech ended ({len(self.speech_frames)} frames)")

        # Save audio to file
        turn_counter = getattr(self, 'turn_counter', 0)
        turn_counter += 1
        self.turn_counter = turn_counter

        audio_file = os.path.join(AUDIO_DIR, f"audio_{turn_counter}.wav")

        # Combine frames
        audio_data = b''.join(self.speech_frames)

        # Save as 16kHz WAV
        wf = wave.open(audio_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.vad_sample_rate)
        wf.writeframes(audio_data)
        wf.close()

        # Reset state
        self.is_speaking = False
        self.speech_frames = []

        # Trigger callback with audio file
        if self.on_speech_end:
            self.on_speech_end(audio_file, turn_counter)

    def get_state(self):
        """Get current recording state"""
        with self.lock:
            return self.is_speaking

    def pause_monitoring(self):
        """Temporarily pause monitoring (e.g., during model response)"""
        if self.monitoring:
            print("[VAD] Pausing monitoring (model responding)")
            self.monitoring = False
            # Only join if we're not calling from within the monitor thread
            if self.monitor_thread and threading.current_thread() != self.monitor_thread:
                self.monitor_thread.join(timeout=1)

    def resume_monitoring(self):
        """Resume monitoring after pause"""
        if not self.monitoring:
            print("[VAD] Resuming monitoring")
            self.monitoring = True
            self.is_speaking = False
            self.speech_frames = []
            self.voice_buffer.clear()
            self.monitor_thread = threading.Thread(target=self._monitor_audio, daemon=True)
            self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

    def close(self):
        self.stop_monitoring()
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
        self.response_complete_event = threading.Event()
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

        # Clear the event - we're waiting for a new response
        self.response_complete_event.clear()

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

    def wait_for_response_complete(self, timeout=30):
        """Wait for the response to be fully processed and spoken"""
        return self.response_complete_event.wait(timeout)


class VideoDisplay:
    def __init__(self, model_interface, whisper_stt, tts_player):
        self.cap = cv2.VideoCapture(0)
        self.model = model_interface
        self.whisper = whisper_stt
        self.tts = tts_player
        self.running = True
        self.ui_state = UIState.IDLE
        self.state_lock = threading.Lock()
        self.processing_lock = threading.Lock()

        # Setup VAD audio recorder with callback
        self.audio = VADAudioRecorder(on_speech_end=self._on_speech_detected)

        print("\n" + "="*60)
        print(" 🎙️  CONTINUOUS VOICE MODE - Just start talking!")
        print("="*60)
        print(" Controls:")
        print("   [SPACE] - Toggle continuous monitoring ON/OFF")
        print("   [q]     - Quit")
        print("\n Starting continuous voice monitoring...")

        # Start continuous monitoring
        self.audio.start_monitoring()
        self.monitoring_active = True

    def _on_speech_detected(self, audio_file, turn_counter):
        """Callback when VAD detects end of speech - runs in background thread"""
        # Run processing in a separate thread to avoid blocking VAD thread
        threading.Thread(target=self._process_speech, args=(audio_file, turn_counter), daemon=True).start()

    def _process_speech(self, audio_file, turn_counter):
        """Process speech in background thread"""
        # CRITICAL: Pause VAD immediately to avoid recording model's response
        self.audio.pause_monitoring()

        # Set state to processing
        with self.state_lock:
            self.ui_state = UIState.PROCESSING

        print(f"\n[Turn {turn_counter}] Processing speech...")

        # Capture current frame - get fresh frame directly from camera
        ret, frame = self.cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame from camera")
            with self.state_lock:
                self.ui_state = UIState.IDLE
            if self.monitoring_active:
                self.audio.resume_monitoring()
            return

        # Save the full resolution frame (not resized)
        image_file = os.path.join(IMAGE_DIR, f"image_{turn_counter}.jpg")
        write_success = cv2.imwrite(image_file, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if not write_success:
            print(f"[ERROR] Failed to write image to {image_file}")
            with self.state_lock:
                self.ui_state = UIState.IDLE
            if self.monitoring_active:
                self.audio.resume_monitoring()
            return

        # Verify image was saved correctly
        img_size = os.path.getsize(image_file) / 1024.0  # KB
        print(f"[DEBUG] Captured frame: {frame.shape[1]}x{frame.shape[0]}, saved as {img_size:.1f} KB")
        print(f"[DEBUG] Image path: {image_file}")

        try:
            # Get audio duration for debug
            with wave.open(audio_file, 'rb') as wf:
                duration = wf.getnframes() / float(wf.getframerate())
            print(f"[Turn {turn_counter}] Audio: {duration:.2f}s")

            # Transcribe audio to text using Whisper
            transcribed_text = self.whisper.transcribe(audio_file)

            # Check if transcription is empty or just whitespace
            if not transcribed_text or not transcribed_text.strip():
                print("[INFO] Transcription empty - skipping model inference")
                # Return to idle and resume monitoring
                with self.state_lock:
                    self.ui_state = UIState.IDLE
                if self.monitoring_active:
                    self.audio.resume_monitoring()
                    print("[INFO] Ready for next input")
                return

            # Verify image file exists before sending
            if not os.path.exists(image_file):
                print(f"[ERROR] Image file does not exist: {image_file}")
                return

            # Read back the image to verify it's valid
            test_img = cv2.imread(image_file)
            if test_img is None:
                print(f"[ERROR] Cannot read saved image: {image_file}")
                return
            print(f"[DEBUG] Verified image readable: {test_img.shape[1]}x{test_img.shape[0]}")

            # Send text + image to model (keeps PROCESSING state)
            print(f"[DEBUG] Sending to model - Text: '{transcribed_text[:50]}...', Image: {image_file}")
            self.model.send_text_and_image(
                text=transcribed_text,
                image_path=image_file
            )

            # CRITICAL: Wait for complete response cycle (model response + TTS playback)
            print("[INFO] Waiting for model response and TTS to complete...")
            if not self.model.wait_for_response_complete(timeout=30):
                print("[WARNING] Timeout waiting for response")

            # Extra buffer to ensure audio is completely done playing
            print("[INFO] Adding buffer time for audio completion...")
            time.sleep(0.8)

        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Return to idle state and resume VAD monitoring
            with self.state_lock:
                self.ui_state = UIState.IDLE

            if self.monitoring_active:
                self.audio.resume_monitoring()
                print("[INFO] Ready for next input")

    def _draw_ui_overlay(self, frame):
        """Draw status overlay on video frame"""
        h, w = frame.shape[:2]

        with self.state_lock:
            state = self.ui_state

        # Only override to RECORDING if we're in IDLE state and actively recording
        # Don't override PROCESSING state
        if state == UIState.IDLE:
            is_recording = self.audio.get_state()
            if is_recording:
                state = UIState.RECORDING

        # Draw status bar at top
        overlay = frame.copy()

        if state == UIState.IDLE:
            # Green bar - ready
            color = (0, 200, 0)
            text = "READY - Listening..."
            cv2.rectangle(overlay, (0, 0), (w, 60), color, -1)
        elif state == UIState.RECORDING:
            # Red pulsing circle - recording
            color = (0, 0, 255)
            text = "RECORDING"
            cv2.rectangle(overlay, (0, 0), (w, 60), color, -1)
            # Pulsing record indicator
            pulse = int(abs(np.sin(time.time() * 4) * 15) + 10)
            cv2.circle(overlay, (30, 30), pulse, (255, 255, 255), -1)
        else:  # PROCESSING
            # Blue bar - processing
            color = (255, 140, 0)
            text = "PROCESSING..."
            cv2.rectangle(overlay, (0, 0), (w, 60), color, -1)
            # Spinning loader
            angle = (time.time() * 180) % 360
            axes = (20, 20)
            cv2.ellipse(overlay, (30, 30), axes, angle, 0, 270, (255, 255, 255), 3)

        # Blend overlay
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        cv2.putText(frame, text, (70, 38), cv2.FONT_HERSHEY_SIMPLEX,
                   0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Draw monitoring status in corner
        if not self.monitoring_active:
            cv2.putText(frame, "PAUSED (Press SPACE)", (w - 300, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        return frame

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Resize and draw UI overlay
            display_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            display_frame = self._draw_ui_overlay(display_frame)

            cv2.imshow('Gemma Live Call', display_frame)

            key = cv2.waitKey(1) & 0xFF

            # Space bar - toggle monitoring
            if key == ord(' '):
                self.monitoring_active = not self.monitoring_active
                if self.monitoring_active:
                    print("\n[INFO] Voice monitoring RESUMED")
                    if not self.audio.monitoring:
                        self.audio.resume_monitoring()
                else:
                    print("\n[INFO] Voice monitoring PAUSED")
                    self.audio.pause_monitoring()

            # Q - quit
            if key == ord('q'):
                self.running = False

        self.cap.release()
        self.audio.close()
        cv2.destroyAllWindows()

def main():
    # Initialize Whisper for speech-to-text
    # Using "base" model for good balance of speed and accuracy
    whisper_stt = WhisperSTT(model_size="base")

    # Initialize TTS
    tts = TTSPlayer()

    # Create model first to get reference
    model = None

    def handle_response(text):
        text = text.strip()
        print(f"Model: {text}")
        # Improve filter to ignore system logs
        if not text: return
        if text.startswith("Model"): return # Initiation line
        if text.startswith("Stream error"):
            print("!! Streaming Error - Request too large?")
            if model:
                model.response_complete_event.set()
            return

        # Speak the response (blocking - waits until done)
        tts.speak(text)

        # Signal that response is complete (TTS has finished)
        if model:
            print("[INFO] Response fully spoken, signaling complete")
            model.response_complete_event.set()

    model = ModelInterface(MODEL_PATH, LIT_BINARY, handle_response)
    video = VideoDisplay(model, whisper_stt, tts)

    try:
        video.run()
    except KeyboardInterrupt:
        pass
    finally:
        model.running = False

if __name__ == "__main__":
    main()
