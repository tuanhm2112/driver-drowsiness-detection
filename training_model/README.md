# Model Training & Export Pipeline

This directory contains the PyTorch deep learning pipeline for driver drowsiness classification, feature engineering, and edge export for Android deployment.

---

## 📂 Directory Structure

```
training_model/
├── dataset.py                # PyTorch Dataset, augmentation, and EAR/MAR calculators
├── train_npz_model.py        # Sequence training script (BiLSTM, scheduler, early stopping)
├── predict_npz.py            # Inference & evaluation script on test splits
├── export_android_assets.py  # PyTorch -> ONNX export & scaler sync for Android
├── requirements.txt          # Python dependencies for training & export
├── face_landmarker.task      # MediaPipe Face Landmarker model asset
└── result/                   # Best trained checkpoint & scaler statistics
    ├── best_model.pt         # Trained PyTorch weights (~64 KB)
    ├── scaler_stats.npz      # Feature normalization parameters (mean, std)
    ├── metrics.json          # Validation & test evaluation records
    └── label_map.json        # Class mapping {"alert": 0, "drowsy": 1}
```

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Training

Train the sequence classifier on the preprocessed `.npz` dataset:

```bash
python train_npz_model.py \
  --data-root "path/to/Dataset_NPZ" \
  --output-dir "result" \
  --sequence-length 60 \
  --batch-size 32 \
  --epochs 30 \
  --learning-rate 0.0001 \
  --dropout 0.3 \
  --early-stopping-patience 5 \
  --early-stopping-metric "loss" \
  --label-smoothing 0.1 \
  --train-noise-std 0.01 \
  --handcrafted-only
```

**Key Arguments:**
- `--sequence-length`: Number of consecutive frames per temporal window (default: 60 frames = 2 seconds at 30 FPS).
- `--handcrafted-only`: Uses 7 handcrafted features (EAR, MAR, Pitch proxy, Brow-eye distance, Gaze distance) for lightweight real-time edge inference.
- `--label-smoothing`: Prevents overconfidence on ambiguous boundary frames.
- `--early-stopping-metric`: Metric monitored by the `ReduceLROnPlateau` scheduler and early stopping checkpointing (`loss`, `accuracy`, or `f1`).

---

### 3. Inference & Evaluation

Evaluate the model on specific test files or splits:

```bash
python predict_npz.py \
  --data-root "path/to/Dataset_NPZ" \
  --input "path/to/Dataset_NPZ/test" \
  --model-dir "result" \
  --sequence-length 60 \
  --handcrafted-only
```

---

### 4. Exporting to Android Assets

Converts the trained PyTorch model into an optimized ONNX graph and exports normalization parameters (`scaler_mean.json`, `scaler_std.json`) directly to the Android app directory:

```bash
python export_android_assets.py \
  --model-dir "result" \
  --android-assets-dir "../android_app/app/src/main/assets"
```

**Exported Assets:**
- `drowsy_model.onnx`: Optimized ONNX model (dynamic batch support, opset 13).
- `scaler_mean.json`: Feature mean array for input standardization.
- `scaler_std.json`: Feature standard deviation array for input standardization.
- `face_landmarker.task`: MediaPipe bundle for facial detection on device.
