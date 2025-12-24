import subprocess
import json
import os
import cv2
import numpy as np
import time

# Create a dummy image
img = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.putText(img, "Test", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
img_path = os.path.abspath("test_image.jpg")
cv2.imwrite(img_path, img)

# Payload
json_payload = json.dumps({
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image", "path": img_path}
    ]
})

print(f"Sending: {json_payload}")

cmd = ["./lit.macos_arm64", "run", "gemma-3n-E2B"]
process = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Read initial prompt
while True:
    line = process.stdout.readline()
    print(f"Init: {line.strip()}")
    if "Start chatting" in line:
        break
    if not line:
        break

# Send JSON
process.stdin.write(json_payload + "\n")
process.stdin.flush()

# Read response
start_time = time.time()
while time.time() - start_time < 10:
    line = process.stdout.readline()
    if line:
        print(f"Response: {line.strip()}")
    if "Model:" in line:
        pass # Keep reading

process.terminate()
