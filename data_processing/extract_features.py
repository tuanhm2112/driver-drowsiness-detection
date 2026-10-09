"""
extract_features.py
====================
Mục đích: Quét qua toàn bộ các clip .mp4 đã được cắt sẵn,
          áp dụng bộ "Lọc Mềm" (Soft Filter) dựa trên file CSV,
          chạy MediaPipe Face Mesh để trích xuất tọa độ 3D (x, y, z)
          của 468 điểm landmark, và lưu kết quả dưới dạng file .npz.

Cấu trúc thư mục đầu vào (Input):
  /content/drive/MyDrive/Dataset_Main_Capstone/
  ├── 01/
  │   ├── 0_clips/
  │   │   ├── alert/         <-- clip .mp4 cắt từ video 0 (alert)
  │   │   ├── drowsy/
  │   │   └── low_vigilance/
  │   ├── 5_clips/
  │   │   ├── alert/
  │   │   ├── drowsy/
  │   │   └── low_vigilance/
  │   ├── 10_clips/
  │   │   ├── alert/
  │   │   ├── drowsy/
  │   │   └── low_vigilance/
  │   ├── 0_filtered.csv
  │   ├── 5_filtered.csv
  │   └── 10_filtered.csv
  ├── 02/
  │   └── ...

Cấu trúc thư mục đầu ra (Output):
  /content/drive/MyDrive/Dataset_NPZ/
  ├── train/
  │   ├── alert/
  │   │   └── user_01_video0.npz   <-- chứa tất cả landmarks của các clip alert của user 01 từ video 0
  │   ├── drowsy/
  │   └── low_vigilance/
  ├── val/
  └── test/

Bên trong mỗi file .npz:
  - data['landmarks']: ma trận shape [N_clips, 90, 468, 3]
  - data['labels']:    mảng nhãn số tương ứng [N_clips] (0=alert, 1=low_vigilance, 2=drowsy)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
from pathlib import Path
from tqdm import tqdm  # Thanh tiến trình đẹp
import urllib.request
import warnings
warnings.filterwarnings('ignore')

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ============================================================
# CẤU HÌNH - Tùy chỉnh đường dẫn phù hợp với môi trường chạy
# ============================================================

INPUT_ROOT = os.getenv("INPUT_ROOT", "./Dataset_Main_Capstone")
OUTPUT_ROOT = os.getenv("OUTPUT_ROOT", "./Dataset_NPZ") 
MODEL_PATH = os.getenv("MODEL_PATH", "face_landmarker.task")

TRAIN_USERS = [str(i) for i in range(49, 54)]  # 55, 56, 57, 58, 59
VAL_USERS   = ["54"]                           # 60
TEST_USERS  = []                               # Tạm thời để trống
# Thông số video
TARGET_FRAMES = 90      # Số frame chuẩn cho 1 clip 3 giây (30 FPS × 3s)
N_LANDMARKS   = 70      # Số điểm landmark được chọn lọc (giảm từ 468 xuống 70)
N_COORDS      = 3       # Số chiều tọa độ (x, y, z)

# ============================================================
# 70 ĐIỂM LANDMARK ĐƯỢC CHỌN LỌC
# Tập trung vào các vùng quan trọng cho bài toán buồn ngủ:
#   - Mắt trái/phải  : Tính EAR (Eye Aspect Ratio)
#   - Lông mày       : Phát hiện nhíu mày, mệt mỏi
#   - Miệng/Môi      : Tính MAR (Mouth Aspect Ratio), phát hiện ngáp
#   - Mũi            : Làm điểm neo, tính chuyển động đầu
#   - Iris           : Hướng nhìn (chỉ có khi refine_landmarks=True)
#   - Viền mặt       : Tư thế đầu (Head Pose)
# ============================================================
KEY_LANDMARKS = [
    # --- Mắt trái (6 điểm - dùng tính EAR) ---
    33, 160, 158, 133, 153, 144,
    # --- Mắt phải (6 điểm - dùng tính EAR) ---
    362, 385, 387, 263, 373, 380,
    # --- Lông mày trái (10 điểm) ---
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    # --- Lông mày phải (10 điểm) ---
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
    # --- Môi/Miệng (14 điểm - dùng tính MAR) ---
    61, 291, 13, 14, 82, 312, 87, 317, 78, 308, 95, 324, 0, 17,
    # --- Mũi (6 điểm - điểm neo và chuyển động đầu) ---
    1, 2, 5, 4, 19, 94,
    # --- Iris trái (5 điểm - hướng nhìn) ---
    468, 469, 470, 471, 472,
    # --- Iris phải (5 điểm - hướng nhìn) ---
    473, 474, 475, 476, 477,
    # --- Viền mặt / Jaw (8 điểm - tư thế đầu) ---
    10, 152, 234, 454, 127, 356, 162, 389,
]
# Tổng: 6+6+10+10+14+6+5+5+8 = 70 điểm ✅

# Ánh xạ nhãn từ tên chuỗi -> số nguyên
# ⭐ Chỉ còn 2 nhãn: alert=0, drowsy=1 (low_vigilance đã gộp vào drowsy)
LABEL_MAP = {
    "alert":  0,
    "drowsy": 1,
}

# ============================================================
# BẢNG QUY TẮC LỌC MỀM (Soft Filter Rules)
# Key   = (tên_folder_video, nhãn_script_detect)
# Value = nhãn_cuối_cùng_được_gán  |  None = VỨT BỎ clip này
# ============================================================
# Giải thích logic:
#   - Video gốc 0  (alert):          Giữ lại alert, vứt drowsy/low_vigilance mâu thuẫn
#   - Video gốc 5  (low_vigilance):  Giữ lại low_vigilance, vứt alert mâu thuẫn, giữ drowsy
#   - Video gốc 10 (drowsy):         Giữ lại drowsy + low_vigilance (gán là drowsy), vứt alert
# ============================================================
SOFT_FILTER_RULES = {
    # (folder_video, nhãn_detect) -> nhãn_cuối
    # ⭐ Chỉ còn 2 nhãn đầu ra: "alert" hoặc "drowsy"
    # ⭐ low_vigilance và drowsy đều được gộp thành "drowsy"

    ("0",  "alert"):          "alert",
    ("0",  "low_vigilance"):  None,     # Mâu thuẫn: video alert nhưng detect buồn ngủ -> VỨT BỎ
    ("0",  "drowsy"):         None,     # Mâu thuẫn: video alert nhưng detect drowsy  -> VỨT BỎ

    ("5",  "alert"):          None,     # Mâu thuẫn: video low_vig nhưng detect tỉnh  -> VỨT BỎ
    ("5",  "low_vigilance"):  "drowsy", # Gộp vào drowsy
    ("5",  "drowsy"):         "drowsy", # Gộp vào drowsy

    ("10", "alert"):          None,     # Mâu thuẫn: video drowsy nhưng detect tỉnh  -> VỨT BỎ
    ("10", "low_vigilance"):  "drowsy", # ⭐ Buồn ngủ ngầm -> gộp vào drowsy
    ("10", "drowsy"):         "drowsy",
}

# ============================================================
# KHỞI TẠO MEDIAPIPE TASK API (phiên bản 0.10+)
# ============================================================
def init_face_landmarker(model_path: str):
    """Tải model và khởi tạo FaceLandmarker theo Task API mới."""
    if not os.path.exists(model_path):
        local_cand = Path("face_landmarker.task")
        if local_cand.exists():
            model_path = str(local_cand)
        else:
            print(f"[DOWNLOAD] Đang tải model MediaPipe về {model_path} ...")
            os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                model_path
            )
            print("[OK] Tải model xong!")

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,  # Xử lý từng frame độc lập
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return vision.FaceLandmarker.create_from_options(options)


# ============================================================
# HÀM XỬ LÝ CHÍNH
# ============================================================

def extract_landmarks_from_clip(video_path: str, landmarker) -> np.ndarray | None:
    """
    Đọc 1 clip .mp4, chạy MediaPipe Task API và trả về ma trận tọa độ.

    Args:
        video_path: Đường dẫn tới file .mp4
        landmarker: Instance FaceLandmarker đã được khởi tạo

    Returns:
        np.ndarray shape [TARGET_FRAMES, N_LANDMARKS, N_COORDS] nếu thành công
        None nếu video bị lỗi hoặc không tìm thấy khuôn mặt > 50% số frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frames_landmarks = []
    no_face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển BGR -> RGB rồi bọc vào mp.Image (Task API yêu cầu)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(mp_image)

        if result.face_landmarks and len(result.face_landmarks) > 0:
            lm = result.face_landmarks[0]  # List các NormalizedLandmark
            # Chỉ lấy 70 điểm quan trọng theo danh sách KEY_LANDMARKS
            coords = np.array(
                [[lm[i].x, lm[i].y, lm[i].z] for i in KEY_LANDMARKS],
                dtype=np.float32
            )  # Shape: [70, 3]
            frames_landmarks.append(coords)
        else:
            no_face_count += 1
            # Dùng frame toàn 0 làm placeholder khi không detect được mặt
            frames_landmarks.append(np.zeros((N_LANDMARKS, N_COORDS), dtype=np.float32))

    cap.release()

    # Nếu quá 50% frame không detect được mặt, coi clip này là rác và bỏ đi
    if len(frames_landmarks) == 0 or no_face_count / max(len(frames_landmarks), 1) > 0.5:
        return None

    frames_landmarks = np.array(frames_landmarks, dtype=np.float32)  # [T, N, 3]

    # --- Chuẩn hóa số lượng frame về đúng TARGET_FRAMES ---
    T = frames_landmarks.shape[0]
    if T >= TARGET_FRAMES:
        # Cắt bớt (lấy đúng 90 frame đầu)
        frames_landmarks = frames_landmarks[:TARGET_FRAMES]
    else:
        # Đệm thêm bằng frame cuối cùng (Last-frame padding)
        pad_size = TARGET_FRAMES - T
        pad = np.repeat(frames_landmarks[-1:], pad_size, axis=0)
        frames_landmarks = np.concatenate([frames_landmarks, pad], axis=0)

    return frames_landmarks  # [TARGET_FRAMES, N_LANDMARKS, N_COORDS]


def get_clips_for_user(user_folder: str) -> dict:
    """
    Lấy tất cả clip hợp lệ của 1 người dùng sau khi áp dụng Soft Filter.
    Tự động quét các folder _clips (bao gồm cả 10_1_clips, 10_2_clips...)
    """
    clips_by_label = {"alert": [], "drowsy": []}
    user_path = Path(user_folder)
    
    # Quét tất cả thư mục kết thúc bằng "_clips"
    clips_dirs = [d for d in user_path.iterdir() if d.is_dir() and d.name.endswith("_clips")]
    
    if not clips_dirs:
        print(f"  ⚠️  Không tìm thấy thư mục _clips nào trong: {user_path}")
        return clips_by_label

    for clips_dir in clips_dirs:
        # Lấy phần đầu (vd: "0", "5", "10_1", "10_2")
        video_prefix = clips_dir.name.replace("_clips", "")
        
        # Đưa về base_type để tra bảng SOFT_FILTER_RULES
        if video_prefix.startswith("10"):
            base_type = "10"
        elif video_prefix.startswith("5"):
            base_type = "5"
        elif video_prefix.startswith("0"):
            base_type = "0"
        else:
            continue

        # Lặp qua các nhãn detect từ script filter (alert, drowsy, low_vigilance)
        for detected_label in ["alert", "low_vigilance", "drowsy"]:
            label_dir = clips_dir / detected_label
            if not label_dir.exists():
                continue

            # Tra cứu bảng Soft Filter dựa vào base_type ("0", "5" hoặc "10")
            rule_key = (base_type, detected_label)
            final_label = SOFT_FILTER_RULES.get(rule_key, None)

            if final_label is None:
                # Bỏ qua (mâu thuẫn)
                continue

            # Gom tất cả file .mp4 trong thư mục này vào nhãn cuối cùng
            mp4_files = list(label_dir.glob("*.mp4"))
            clips_by_label[final_label].extend(mp4_files)

    return clips_by_label


def process_and_save_user(user_id: str, split: str, landmarker):
    """
    Xử lý toàn bộ clips của 1 người dùng và lưu thành file .npz.
    
    Args:
        user_id: Ví dụ "01", "02", ...
        split:   "train", "val", hoặc "test"
    """
    user_folder = os.path.join(INPUT_ROOT, user_id)
    if not os.path.exists(user_folder):
        print(f"❌ Không tìm thấy folder người dùng: {user_folder}")
        return

    print(f"\n{'='*60}")
    print(f"📁 Đang xử lý User {user_id} -> [{split}]")

    # Lấy danh sách clip sau khi lọc
    clips_by_label = get_clips_for_user(user_folder)

    for label_name, clip_paths in clips_by_label.items():
        if len(clip_paths) == 0:
            print(f"  ℹ️  Nhãn [{label_name}]: Không có clip nào (bị lọc hết).")
            continue

        print(f"  🎬 Nhãn [{label_name}]: Đang trích xuất {len(clip_paths)} clips...")

        all_landmarks = []
        all_labels    = []
        skipped       = 0

        for clip_path in tqdm(clip_paths, desc=f"    User {user_id} | {label_name}", leave=False):
            landmarks = extract_landmarks_from_clip(str(clip_path), landmarker)

            if landmarks is None:
                skipped += 1
                continue

            all_landmarks.append(landmarks)
            all_labels.append(LABEL_MAP[label_name])

        if len(all_landmarks) == 0:
            print(f"  ⚠️  Nhãn [{label_name}]: 0 clip hợp lệ sau khi xử lý MediaPipe (bỏ qua).")
            continue

        # Tạo mảng NumPy cuối cùng
        landmarks_array = np.array(all_landmarks, dtype=np.float32)  # [N_clips, 90, 468, 3]
        labels_array    = np.array(all_labels,    dtype=np.int64)    # [N_clips]

        # Tạo thư mục đích nếu chưa có
        out_dir = Path(OUTPUT_ROOT) / split / label_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Lưu file .npz
        out_path = out_dir / f"user_{user_id}_video.npz"
        np.savez_compressed(
            str(out_path),
            landmarks=landmarks_array,
            labels=labels_array
        )

        print(f"  ✅ Đã lưu: {out_path.name} | Shape: {landmarks_array.shape} | Bỏ qua: {skipped} clips lỗi")


# ============================================================
import concurrent.futures

def process_user_worker(args):
    """Hàm bọc để mỗi luồng tự khởi tạo model MediaPipe riêng"""
    user_id, split_name = args
    landmarker = init_face_landmarker(MODEL_PATH)
    try:
        process_and_save_user(user_id, split_name, landmarker)
    finally:
        landmarker.close()
    return user_id

def run():
    print("🚀 Bắt đầu trích xuất MediaPipe (Chế độ MULTIPROCESSING - Bung 100% CPU)...")
    print(f"   Input : {INPUT_ROOT}")
    print(f"   Output: {OUTPUT_ROOT}")
    print(f"   Train : {len(TRAIN_USERS)} người | Val: {len(VAL_USERS)} người | Test: {len(TEST_USERS)} người")
    splits = [
        ("train", TRAIN_USERS),
        ("val",   VAL_USERS),
        ("test",  TEST_USERS),
    ]

    # Sử dụng 5 nhân CPU (Tối đa công suất của instance)
    MAX_WORKERS = 5

    for split_name, user_list in splits:
        print(f"\n{'#'*60}")
        print(f"# Đang xử lý tập [{split_name.upper()}] bằng {MAX_WORKERS} CPU Cores song song")
        print(f"{'#'*60}")

        # Đóng gói công việc
        tasks = [(user_id, split_name) for user_id in user_list]
        
        # Chạy đa luồng
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(process_user_worker, task) for task in tasks]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    user_done = future.result()
                except Exception as exc:
                    print(f"❌ Lỗi khi xử lý luồng: {exc}")

    print("\n" + "="*60)
    print("🎉 HOÀN THÀNH TOÀN BỘ!")
    print(f"   {OUTPUT_ROOT}")

if __name__ == "__main__":
    run()
