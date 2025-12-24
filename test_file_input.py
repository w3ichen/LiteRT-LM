import subprocess
import json
import os
import cv2
import numpy as np

# Create dummy media
img_path = os.path.abspath("test_input.jpg")
cv2.imwrite(img_path, np.zeros((100, 100, 3), dtype=np.uint8))

audio_path = os.path.abspath("test_input.wav")
# Create dummy wav file
import wave
with wave.open(audio_path, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(16000)
    wf.writeframes(b'\x00' * 32000) # 2 seconds of silence

# Create prompt file
prompt_content = json.dumps({
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the image and audio."},
        {"type": "image", "path": img_path},
         {"type": "audio", "path": audio_path}
    ]
})

prompt_file = "input_prompt.json"
with open(prompt_file, "w") as f:
    f.write(prompt_content)

print(f"Created prompt file: {prompt_file}")
print(f"Content: {prompt_content}")

cmd = ["./lit.macos_arm64", "run", "gemma-3n-E4B", "--backend", "cpu", "-f", prompt_file]
print(f"Running: {' '.join(cmd)}")

subprocess.run(cmd)
