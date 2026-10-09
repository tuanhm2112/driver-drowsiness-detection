# Data Processing & Feature Extraction Pipeline

This directory contains the preprocessing pipeline for preparing driving video datasets for sequence-based drowsiness classification.

---

## 📋 Pipeline Architecture

```
Raw Driving Videos (MP4)
       │
       ▼
1. filter_video.py
   ├── MediaPipe Face Landmarker
   ├── Rule-based Metrics: EAR, MAR, Head Pitch Proxy, PERCLOS
   └── Soft-Filtering & Slicing into 3-second clips (.mp4)
       │
       ▼
2. extract_features.py
   ├── 70 Selected Key Facial Landmarks (Eyes, Brows, Mouth, Nose, Irises)
   ├── Spatial Normalization (Nose Anchor + Inter-ocular distance)
   └── Multiprocessing export to compressed .npz archives
       │
       ▼
Structured NPZ Dataset (train / val / test)
```

---

## 🛠️ Components

### 1. `filter_video.py`
Analyzes long driver recordings frame-by-frame, computes eye closure and yawning statistics, and segments continuous recordings into labeled 3-second candidate clips (`alert`, `drowsy`, `low_vigilance`).

**Key Detection Metrics:**
- **EAR (Eye Aspect Ratio):** Detects blinks and prolonged eye closure.
- **MAR (Mouth Aspect Ratio):** Identifies yawning episodes.
- **Head Pitch Proxy:** Detects forward head dropping / nodding.
- **PERCLOS:** Percentage of eyelid closure over a sliding time window.

**Usage:**
```bash
python filter_video.py \
  --input-video "path/to/raw_driver_video.mp4" \
  --output-dir "path/to/clips_output" \
  --min-confidence 0.5
```

---

### 2. `extract_features.py`
Processes segmented `.mp4` video clips in parallel across multiple CPU cores, tracks 70 curated landmarks from MediaPipe Face Landmarker, applies label alignment rules, and packages the data into `.npz` files ready for deep learning training.

**70 Selected Landmark Topology:**
- **Eyes (12 points):** 6 points per eye for precise EAR calculation.
- **Eyebrows (20 points):** 10 points per brow for fatigue and frown tracking.
- **Mouth / Lips (14 points):** Outer and inner lips for yawning dynamics (MAR).
- **Nose Tip & Bridge (6 points):** Stable coordinate anchor and head translation proxy.
- **Irises (10 points):** 5 points per eye for gaze direction monitoring.
- **Jaw / Contour (8 points):** Head roll and tilt estimation.

**Output NPZ Data Format:**
```python
data = np.load("user_XX_video.npz")
landmarks = data["landmarks"]  # Shape: [N_clips, 90, 70, 3] (T=90 frames, 70 points, (x, y, z))
labels    = data["labels"]     # Shape: [N_clips] (0: alert, 1: drowsy)
```

**Usage:**
```bash
python extract_features.py
```
*(Environment variables `INPUT_ROOT`, `OUTPUT_ROOT`, and `MODEL_PATH` can be set to configure custom paths).*

---

## 📦 Model Asset
- `face_landmarker.task`: MediaPipe Face Landmarker bundle (Float16). The script will automatically fetch it from Google Storage CDN if not present locally.
