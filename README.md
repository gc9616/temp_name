# HemiSec — Palm Vein Biometric Capture (Raspberry Pi)

A hardware + software prototype for **palm-vein biometric capture** using near-IR imaging on a Raspberry Pi. The system captures a palm image under NIR illumination and runs a feature-extraction pipeline, then forwards the image to a server for downstream comparison.

## Elevator pitch
**Fast, IR-based palm-vein capture on a Raspberry Pi for lightweight biometric identity verification.**

---

## What this repo is
This repo contains:
- **Raspberry Pi capture software** (camera → image files and/or API endpoint)
- A **palm-vein feature extraction pipeline** (segmentation → illumination correction → filter-bank response → feature vector)
- A simple **Flask API** (capture request → returns image or derived feature vector)
---

## Hardware overview

### Core components
- **Raspberry Pi 3 (32-bit OS)**  
  - Constraint: CPU-only, limited RAM; heavy OpenCV pipelines can be slow at full resolution.
- **Arducam (NOIR)** camera module  
  - NIR = no IR-cut filter → works well with near-IR illumination.
- **External IR illumination panel**
  - Near-IR LED panel (commonly 850nm; 940nm also possible but often dimmer on typical sensors).
- **Battery power**
  - **12V battery pack**: 8× AA in a battery case providing 12 volts of power.
  - Battery power soldered to 850 nm light.
- **3D prints**
   - Housing for electronics and phone designed in Solidworks and printed with 8% infill PLA


---

## Software overview

### High-level flow
1. **Capture** an image on the Pi (libcamera `rpicam-still`)
2. **Preprocess + segment** the hand/palm region (mask + “safe mask” away from edges)
3. **Illumination correction** to reduce global lighting effects
4. **Feature extraction** (e.g., oriented filter-bank response pooled into a vector)
5. Either:
   - return the feature vector (fast matching), or
   - return the image (server-side extraction/matching)

---

## Raspberry Pi set up

### 1) Enable camera + install system deps
On Raspberry Pi OS:
```bash
sudo apt update
sudo apt install -y libcamera-apps python3 python3-venv python3-pip

## 2) Create a virtual environment

```bash
cd ~/hemisec
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

## 3) Install Python requirements

```bash
pip install -r requirements.txt
```

deps:
- `numpy`
- `opencv-python` (or `python3-opencv` via apt)
- `scikit-image` (only if you rely on it; avoid if you want speed)
- `flask`
---

## Capturing images (CLI)

### Test capture

```bash
mkdir -p palm_images
rpicam-still -o palm_images/test.jpg --width 2592 --height 1944 --nopreview
```

## Flask API (Pi-side)

This repo includes a small Flask app that exposes a capture endpoint.

### Endpoints
- `GET /health` → service health check
- `POST /capture` → triggers capture and returns payload


## Suggested workflow for reliable capture

1. Turn on IR panel, confirm camera exposure isn’t saturating.
2. Have user place palm in a consistent pose (physical guide).
3. Capture multiple frames; use the sharpest/least saturated (optional improvement).
4. Extract vector (Pi or server).
5. Compare vectors using cosine similarity (or your matcher of choice).

---

## Feature extraction pipeline

> Goal: turn a near-IR palm image into a **stable feature vector**

### 1) Hand segmentation + safe mask
**Purpose:** isolate the hand region and avoid the boundary where silhouette artifacts dominate.

Typical outputs:
- `hand_mask`: binary blob of the hand
- `safe_mask`: eroded/core region that stays away from edges (used everywhere downstream)

Segmentation:
- Blur → Otsu threshold
- Flood fill from borders to remove background regions connected to frame edges
- Largest/most central component selection
- Morph close/open + hole fill for cleanup
- Distance transform to generate `safe_mask` (avoid boundary band artifacts)

### 2) Illumination correction
**Purpose:** remove slow-varying lighting so the algorithm sees veins as local contrast features rather than brightness gradients.

- **“Full” illumination correction**: background blur → subtract → normalize → CLAHE → blur  
  - Pros: makes veins visually pop
  - Cons: CLAHE can introduce nonlinear contrast changes; can hurt descriptor invariance

- **“For features” correction**:
  - background blur → subtract → robust percentile normalization inside `safe_mask`
  - no CLAHE
  - more photometrically stable for feature vectors

### 3) Gabor filter bank (vein/ridge response)
**Purpose:** capture oriented “ridge/line” energy in the palm.

### 4) Building the feature vector

- **Per-orientation response stack** 
- **Spatial pooling** (grid pooling) to tolerate small translations
- **Normalization** (L2 or local block norm) to reduce contrast/illumination sensitivity

- Divide ROI into `grid × grid` cells
- For each cell and each orientation, compute pooled energy (mean or RMS)
- Normalize per-cell across orientations (gain invariance)
- Optionally HOG-style block normalization
- Final global L2 normalization → 1D feature vector
