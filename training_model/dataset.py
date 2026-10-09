import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CHỈ SỐ VỊ TRÍ TRONG 70 ĐIỂM LANDMARK
# ============================================================
NOSE_TIP_IDX  = 46
LEFT_EYE_IDX  = 0
RIGHT_EYE_IDX = 6

# Mắt trái: indices 0-5  (points: 33, 160, 158, 133, 153, 144)
L_EYE = [0, 1, 2, 3, 4, 5]
# Mắt phải: indices 6-11 (points: 362, 385, 387, 263, 373, 380)
R_EYE = [6, 7, 8, 9, 10, 11]
# Miệng: indices 32-45
# 32=left_corner, 33=right_corner, 44=top_lip, 45=bottom_lip
# 34=top_inner, 35=bottom_inner, 36=top_left, 38=bottom_left
MOUTH_CORNER_L = 32
MOUTH_CORNER_R = 33
MOUTH_TOP      = 44
MOUTH_BOT      = 45
MOUTH_TOP_L    = 36
MOUTH_BOT_L    = 38
MOUTH_TOP_R    = 37
MOUTH_BOT_R    = 39


def eye_aspect_ratio(eye_pts):
    """
    Tính EAR (Eye Aspect Ratio) cho 1 mắt.
    eye_pts: array shape [6, 2] - 6 điểm mắt, chỉ lấy x,y
    EAR = (||p1-p5|| + ||p2-p4||) / (2 * ||p0-p3||)
    Giá trị ~0.3 lúc mắt mở, tiến về 0 khi nhắm mắt.
    """
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def mouth_aspect_ratio(pts):
    """
    Tính MAR (Mouth Aspect Ratio).
    MAR = chiều cao miệng / chiều rộng miệng
    Tăng vọt khi ngáp, xấp xỉ 0 khi môi khép.
    """
    height = (np.linalg.norm(pts[MOUTH_TOP]   - pts[MOUTH_BOT]) +
              np.linalg.norm(pts[MOUTH_TOP_L]  - pts[MOUTH_BOT_L]) +
              np.linalg.norm(pts[MOUTH_TOP_R]  - pts[MOUTH_BOT_R])) / 3.0
    width  = np.linalg.norm(pts[MOUTH_CORNER_L] - pts[MOUTH_CORNER_R])
    return height / (width + 1e-6)


def compute_ear_mar(x: np.ndarray) -> np.ndarray:
    """
    Tính EAR trung bình và MAR cho từng frame trong clip.

    Args:
        x: [90, 70, 3] - tọa độ (x, y, z) của 70 điểm qua 90 frames

    Returns:
        ear_mar: [90, 2] - (EAR_avg, MAR) cho từng frame
    """
    T = x.shape[0]
    ear_mar = np.zeros((T, 2), dtype=np.float32)

    for t in range(T):
        frame = x[t]  # [70, 3]
        xy    = frame[:, :2]  # Chỉ dùng x, y (2D) để tính khoảng cách

        ear_l = eye_aspect_ratio(xy[L_EYE])
        ear_r = eye_aspect_ratio(xy[R_EYE])
        ear_avg = (ear_l + ear_r) / 2.0

        mar = mouth_aspect_ratio(xy)

        ear_mar[t, 0] = ear_avg
        ear_mar[t, 1] = mar

    return ear_mar  # [90, 2]


def spatial_normalize(x: np.ndarray) -> np.ndarray:
    """
    Chuẩn hóa Tọa độ Không gian với 3 lớp bảo vệ chống NaN/Inf.
    """
    if np.max(np.abs(x)) < 1e-6:
        return x

    origin = x[0, NOSE_TIP_IDX, :]
    x_norm = x - origin[np.newaxis, np.newaxis, :]

    left_eye  = x_norm[0, LEFT_EYE_IDX, :2]
    right_eye = x_norm[0, RIGHT_EYE_IDX, :2]
    eye_distance = np.linalg.norm(left_eye - right_eye)

    if eye_distance < 0.01:
        mid_frame = x_norm[45, :, :]
        left_eye  = mid_frame[LEFT_EYE_IDX, :2]
        right_eye = mid_frame[RIGHT_EYE_IDX, :2]
        eye_distance = np.linalg.norm(left_eye - right_eye)

    if eye_distance < 0.01:
        return np.clip(x_norm, -5.0, 5.0)

    x_norm = x_norm / eye_distance
    x_norm = np.clip(x_norm, -5.0, 5.0)
    return x_norm


class DrowsinessDataset(Dataset):
    def __init__(self, data_dir, split="train", augment=False):
        self.data_dir = Path(data_dir) / split
        self.augment  = augment
        self.samples  = []
        self.labels   = []

        for label_name, label_idx in [("alert", 0), ("drowsy", 1)]:
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                continue
            for npz_file in label_dir.glob("*.npz"):
                data      = np.load(npz_file)
                landmarks = data['landmarks']  # [N, 90, 70, 3]
                lbls      = data['labels']     # [N]
                self.samples.extend(landmarks)
                self.labels.extend(lbls)

        print(f"Đã load tập {split}: {len(self.samples)} clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx].copy()  # [90, 70, 3]
        y = self.labels[idx]

        # --- Bước 1: Tính EAR và MAR TRƯỚC khi normalize ---
        # (Vì EAR/MAR dùng tọa độ tuyệt đối từ MediaPipe, đã nhất quán giữa các người)
        ear_mar = compute_ear_mar(x)  # [90, 2]

        # --- Bước 2: Chuẩn hóa không gian ---
        x = spatial_normalize(x)     # [90, 70, 3]

        # --- Bước 3: Augmentation ---
        if self.augment:
            # [1] Lật gương trục X (mô phỏng camera đặt bên trái/phải)
            if np.random.rand() > 0.5:
                x[:, :, 0]    = -x[:, :, 0]
                # EAR/MAR là tỷ lệ → không đổi khi lật gương

            # [2] Gaussian Noise (mô phỏng camera chất lượng thấp, run tay nhẹ)
            if np.random.rand() > 0.5:
                x       += np.random.normal(0, 0.002, x.shape)
                ear_mar += np.random.normal(0, 0.005, ear_mar.shape)

            # [3] Scale Jitter (mô phỏng người ngồi gần/xa camera)
            # EAR/MAR là tỷ lệ nên không bị ảnh hưởng khi scale tọa độ
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.95, 1.05)
                x = x * scale

            # [4] Temporal Shift — TẮT: gây gradient quá nhiễu với 48 người
            # [5] Frame Masking  — TẮT: phân phối train/val quá khác nhau

        return (
            torch.tensor(x,       dtype=torch.float32),   # [90, 70, 3]
            torch.tensor(ear_mar, dtype=torch.float32),   # [90, 2]
            torch.tensor(y,       dtype=torch.long)
        )


def get_dataloaders(data_root, batch_size=32):
    train_dataset = DrowsinessDataset(data_root, split="train", augment=True)
    val_dataset   = DrowsinessDataset(data_root, split="val",   augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    return train_loader, val_loader
