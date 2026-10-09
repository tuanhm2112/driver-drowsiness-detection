# Real-Time Driver Drowsiness Android App

A native Android application providing real-time driver fatigue and drowsiness detection using on-device computer vision and edge machine learning.

---

## 📱 Features

- **Real-Time Video Analysis:** Powered by **CameraX** at 30 FPS with front-facing camera input.
- **On-Device Facial Landmark Extraction:** Uses **Google MediaPipe Face Landmarker** (Task API) running fully offline.
- **Handcrafted Fatigue Metrics:** Real-time calculation of Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pitch Proxy.
- **Edge ML Inference:** Runs the quantized sequence **BiLSTM ONNX model** via **Microsoft ONNX Runtime Android** for sub-10ms inference latency.
- **Personalized Calibration:** Baseline calibration step during startup to adapt to individual facial geometry and camera placement angles.
- **Multi-Level Warning System:**
  - `TỈNH TÁO` (Alert): Green status HUD.
  - `CẦN CHÚ Ý` (Caution): Orange visual warning banner.
  - `PHÁT HIỆN BUỒN NGỦ!` (Drowsy): Red warning with alarm sound and haptic vibration pattern.
- **Picture-in-Picture (PiP):** Supports PiP mode to keep monitoring drivers while they use navigation apps (Google Maps, Waze).

---

## 🏗️ Architecture

```
Front Camera (CameraX)
       │
       ▼ (30 FPS Stream)
MediaPipe Face Landmarker (468 landmarks)
       │
       ▼
FeatureCalculator.kt
  ├── EAR (Left & Right Eye Aspect Ratio)
  ├── MAR (Mouth Aspect Ratio)
  └── Head Pitch Proxy
       │
       ▼
SlidingWindowBuffer.kt (60-frame temporal window)
       │
       ▼
DrowsinessClassifier.kt (ONNX Runtime)
  ├── Standardized with scaler_mean.json & scaler_std.json
  └── BiLSTM model inference
       │
       ▼
AlertManager.kt
  ├── Visual HUD banner update
  ├── Audio alert (Ringtone / MediaPlayer)
  └── Haptic vibration feedback
```

---

## 🛠️ Tech Stack & Dependencies

- **Language:** Kotlin
- **Camera:** Android Jetpack CameraX (`1.3.4`)
- **Vision AI:** Google MediaPipe Tasks Vision (`0.10.14`)
- **Model Inference:** Microsoft ONNX Runtime Android (`1.18.0`)
- **UI:** Material Components, ConstraintLayout, Picture-in-Picture API
- **Minimum SDK:** 24 (Android 7.0 Nougat)
- **Target SDK:** 36 (Android 15)

---

## 📦 Assets

Pre-bundled inside `app/src/main/assets/`:
- `drowsy_model.onnx`: Optimized sequence model exported from PyTorch.
- `face_landmarker.task`: MediaPipe Face Mesh model bundle.
- `scaler_mean.json`: Input feature mean array for Z-score normalization.
- `scaler_std.json`: Input feature standard deviation array.

---

## 🚀 Building & Running

### Using Android Studio
1. Open Android Studio.
2. Select **Open** and choose the `android_app` directory.
3. Allow Gradle to sync dependencies.
4. Connect an Android device (via USB with USB Debugging enabled) or start an Emulator.
5. Click **Run 'app'** (`Shift + F10`).

### Using Gradle CLI
```bash
# On Windows
./gradlew.bat assembleDebug

# On macOS/Linux
./gradlew assembleDebug
```

The compiled APK will be located at:
`app/build/outputs/apk/debug/app-debug.apk`
