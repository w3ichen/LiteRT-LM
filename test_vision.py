#!/usr/bin/env python3
"""Quick test to verify vision input to model"""
import subprocess
import json
import os

# Use the most recent image
image_path = os.path.abspath("image_inputs/image_1.jpg")

if not os.path.exists(image_path):
    print(f"Image not found: {image_path}")
    print("Please run live_call.py first to generate test images")
    exit(1)

# Create test prompt
message = {
    "role": "user",
    "content": [
        {"type": "image", "path": image_path},
        {"type": "text", "text": "Describe in detail what you see in this image. Be specific about objects, people, setting, colors, and any text visible."}
    ]
}

prompt_file = "test_vision_prompt.json"
with open(prompt_file, "w") as f:
    json.dump(message, f, indent=2)

print(f"Testing vision with image: {image_path}")
print(f"Image size: {os.path.getsize(image_path) / 1024:.1f} KB")
print("\nSending to model...\n")

# Run model
cmd = ["./lit.macos_arm64", "run", "gemma-3n-E2B", "--backend", "gpu", "-f", prompt_file]
subprocess.run(cmd)

# Cleanup
os.remove(prompt_file)
