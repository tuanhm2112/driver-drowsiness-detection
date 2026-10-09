# Model Benchmark & Comparative Analysis

This directory contains experimental results comparing sequence deep learning architectures for driver drowsiness detection.

All models were trained on 60-frame sequences with handcrafted facial features (EAR, MAR, Head Pitch Proxy, Brow Distance, Iris Gaze).

| Model Architecture | Best Epoch | Val Acc (%) | Val F1 (%) | Test Acc (%) | Test F1 (%) | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM + Attention (Handcrafted)** | 6 | 92.01% | 91.29% | **91.70%** | **91.21%** | 0.2301 |
| **BiLSTM (Handcrafted - Android Deployment)** | 6 | 92.81% | 92.21% | **90.30%** | **89.52%** | 0.2542 |
| **BiLSTM Baseline (NPZ)** | 8 | 92.26% | 91.61% | **90.26%** | **89.54%** | 0.2510 |
| **Vanilla LSTM (Handcrafted)** | 5 | 93.21% | 92.78% | **89.90%** | **89.10%** | 0.2679 |
| **TCN (Temporal Convolutional Network)** | 7 | 89.13% | 87.24% | **88.49%** | **86.94%** | 0.2943 |
| **GNN-TCN (Graph Neural Network + TCN)** | 5 | 88.97% | 87.12% | **87.11%** | **85.05%** | 0.3187 |

## Key Findings

1. **LSTM + Self-Attention** achieved the highest test generalization (**91.70% Accuracy**, **91.21% F1**), successfully attending to critical eye-closure and head-nodding transition moments.
2. **BiLSTM** achieved **90.30% Test Accuracy** with minimal footprint (63 KB PyTorch weights, 28 KB ONNX), offering the best balance of speed and accuracy for real-time mobile execution on Android.
3. **Recurrent Models (LSTM / BiLSTM)** showed superior temporal state tracking over pure convolutional architectures (TCN) for 2-3 second temporal facial dynamic windows.

## Directory Contents

- `outputs_*/best_model.pt`: Checkpoint weights of the best validation epoch.
- `outputs_*/scaler_stats.npz`: Feature normalization statistics (`mean` and `std`).
- `outputs_*/metrics.json`: Detailed validation and test metrics across all epochs.
- `outputs_*/training_curves.png`: Training/validation loss and accuracy curves.
