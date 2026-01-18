#!/usr/bin/env python3

from flask import Flask, jsonify
import io
import base64
from picamera import PiCamera
from time import sleep

app = Flask(__name__)

# -----------------------------
# Capture a single image
# -----------------------------
@app.route("/capture", methods=["POST"])
def capture():
    try:
        # Create camera object (this does not start video streaming)
        with PiCamera() as camera:
            camera.resolution = (640, 480)
            camera.framerate = 30
            sleep(0.5)  # short warm-up time

            # Capture to memory buffer
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg')
            stream.seek(0)
            image_bytes = stream.read()

        # Encode image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        return jsonify({
            "status": "ok",
            "image_base64": image_base64
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Health check
# -----------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("🟢 Pi Capture API (single snapshot) running")
    app.run(host="0.0.0.0", port=8443)
