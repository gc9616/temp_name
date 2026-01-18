#!/usr/bin/env python3

from flask import Flask, jsonify
import subprocess
import base64
import time

# -----------------------------
# Configuration
# -----------------------------
CAPTURE_PATH = "/tmp/capture.jpg"

# -----------------------------
# Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Capture image helper
# -----------------------------
def capture_image():
    """Use raspistill to take a single snapshot and return base64-encoded JPEG"""
    try:
        # Take snapshot without preview (-n), set width/height
        subprocess.run([
            "rpicam-still",
            "-o", CAPTURE_PATH,
            "-n",        # no preview
            "-t", "500"  # 0.5s delay to let camera adjust
        ], check=True)

        # Read file and encode as base64
        with open(CAPTURE_PATH, "rb") as f:
            image_bytes = f.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return image_base64

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"raspistill failed: {e}")
    except Exception as e:
        raise RuntimeError(f"Capture error: {e}")

# -----------------------------
# API Endpoints
# -----------------------------
@app.route("/")
def root():
    return jsonify({
        "message": "Palm-Vein Biometric Capture API",
        "endpoints": {
            "/capture": "POST: Capture a single palm-vein image and return base64-encoded JPEG"
        }
    })

@app.route("/capture", methods=["POST"])
def capture():
    try:
        image_base64 = capture_image()
        return jsonify({
            "status": "ok",
            "biometric_image": image_base64
        })
    except RuntimeError as e:
        return jsonify({"status": "error", "error": str(e)}), 503
    except Exception as e:
        return jsonify({"status": "error", "error": f"Server error: {str(e)}"}), 500

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    print("🟢 Palm-Vein Capture API running (raspistill)")
    print("📸 POST /capture to get a base64 image")
    app.run(host="0.0.0.0", port=8443, threaded=True)
