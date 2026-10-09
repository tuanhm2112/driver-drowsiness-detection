import argparse
import io
import json
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch

# Fix Windows console UTF-8 output if possible
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from train_npz_model import BiLSTMClassifier

LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"


def parse_args():
    parser = argparse.ArgumentParser(description="Export trained PyTorch model and scaler stats to Android assets.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("result"),
        help="Directory containing best_model.pt and scaler_stats.npz (default: result/)",
    )
    parser.add_argument(
        "--android-assets-dir",
        type=Path,
        default=Path("../android_app/app/src/main/assets"),
        help="Target Android assets directory (default: ../android_app/app/src/main/assets)",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=13,
        help="ONNX opset version (default: 13)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=7,
        help="Number of handcrafted features (default: 7)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=60,
        help="Temporal sequence window length (default: 60)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=32,
        help="BiLSTM hidden size (default: 32)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="BiLSTM layer count (default: 1)",
    )
    return parser.parse_args()


def export_assets(args):
    model_dir = args.model_dir
    if not (model_dir / "best_model.pt").exists():
        fallback_dir = Path("../compare_models/outputs_npz_handcrafted_seq60")
        if (fallback_dir / "best_model.pt").exists():
            print(f"[WARN] Model not found in {model_dir}, using fallback: {fallback_dir}")
            model_dir = fallback_dir
        else:
            raise FileNotFoundError(f"Cannot find best_model.pt in {model_dir} or {fallback_dir}")

    android_assets_dir = args.android_assets_dir
    android_assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Source model directory: {model_dir.resolve()}")
    print(f"[INFO] Target assets directory: {android_assets_dir.resolve()}")

    # 1. Export PyTorch model to ONNX
    print("\n1. Exporting PyTorch model to ONNX format...")
    model = BiLSTMClassifier(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=0.3,
        num_classes=2,
    )
    state_dict = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    onnx_path = android_assets_dir / "drowsy_model.onnx"
    dummy_input = torch.randn(1, args.sequence_length, args.input_size)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=args.opset_version,
    )
    print(f"   [OK] Saved ONNX model: {onnx_path}")

    # 2. Extract scaler stats
    scaler_path = model_dir / "scaler_stats.npz"
    if scaler_path.exists():
        print("\n2. Extracting scaler stats (mean, std) to JSON...")
        scaler = np.load(scaler_path)
        mean = scaler["mean"].astype(float).tolist()
        std = scaler["std"].astype(float).tolist()

        with open(android_assets_dir / "scaler_mean.json", "w", encoding="utf-8") as f:
            json.dump(mean, f, indent=2)
        with open(android_assets_dir / "scaler_std.json", "w", encoding="utf-8") as f:
            json.dump(std, f, indent=2)
        print("   [OK] Saved scaler_mean.json & scaler_std.json")
    else:
        print(f"   [WARN] {scaler_path} not found! Skipping scaler export.")

    # 3. MediaPipe face_landmarker.task
    print("\n3. Ensuring face_landmarker.task exists in assets...")
    dest_landmarker = android_assets_dir / "face_landmarker.task"
    if not dest_landmarker.exists():
        src_candidates = [
            Path("face_landmarker.task"),
            Path("../data_processing/process_raw_dataset/face_landmarker.task"),
            Path("../data_processing/face_landmarker.task"),
        ]
        found = False
        for src in src_candidates:
            if src.exists():
                shutil.copy(src, dest_landmarker)
                print(f"   [OK] Copied from {src}")
                found = True
                break
        if not found:
            print(f"   [DOWNLOAD] Downloading face_landmarker.task from CDN...")
            urllib.request.urlretrieve(LANDMARKER_URL, str(dest_landmarker))
            print("   [OK] Downloaded successfully.")
    else:
        print("   [OK] face_landmarker.task already exists in Android assets.")

    print("\n[SUCCESS] Export completed successfully! All assets are synchronized for Android deployment.")


def main():
    args = parse_args()
    export_assets(args)


if __name__ == "__main__":
    main()
