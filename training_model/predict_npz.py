import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch

from train_npz_model import (
    BiLSTMClassifier,
    build_handcrafted_features,
    build_metrics_from_predictions,
    build_user_baselines,
    choose_feature_array,
    choose_label_array,
    coerce_sequence_batch,
    normalize_landmark_batch,
    normalize_sequences,
    personalize_handcrafted_features,
    resize_sequence_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict labels from NPZ files.")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset root containing train/val/test.")
    parser.add_argument("--input", type=Path, required=True, help="NPZ file or directory to evaluate.")
    parser.add_argument("--model-dir", type=Path, required=True, help="Directory with best_model.pt and scaler_stats.npz.")
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--append-ear-mar", action="store_true")
    parser.add_argument("--personalize-handcrafted", action="store_true")
    parser.add_argument("--handcrafted-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=None, help="Override binary threshold.")
    return parser.parse_args()


def infer_split_dir(input_path: Path, data_root: Path) -> Path:
    resolved_input = input_path.resolve()
    resolved_root = data_root.resolve()
    for split_name in ["train", "val", "test"]:
        split_dir = (resolved_root / split_name).resolve()
        try:
            resolved_input.relative_to(split_dir)
            return split_dir
        except ValueError:
            continue
    raise ValueError("Input path must be inside train/, val/, or test/ under data-root.")


def collect_npz_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(input_path.rglob("*.npz"))


def load_threshold(model_dir: Path) -> float | None:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold_info = payload.get("threshold_optimization")
    if not threshold_info:
        return None
    return float(threshold_info.get("best_threshold", 0.5))


def load_model_bundle(model_dir: Path) -> Tuple[Dict[str, int], np.ndarray, np.ndarray, torch.nn.Module]:
    label_map = json.loads((model_dir / "label_map.json").read_text(encoding="utf-8"))
    scaler = np.load(model_dir / "scaler_stats.npz")
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
    feature_dim = int(metrics["feature_dim"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        input_size=feature_dim,
        hidden_size=32,
        num_layers=1,
        dropout=0.3,
        num_classes=len(label_map),
    ).to(device)
    state_dict = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return label_map, mean, std, model


def build_sequences_for_file(
    file_path: Path,
    split_dir: Path,
    sequence_length: int,
    append_ear_mar: bool,
    personalize_handcrafted: bool,
    handcrafted_only: bool,
) -> Tuple[np.ndarray, np.ndarray | None]:
    user_baselines = build_user_baselines(split_dir) if append_ear_mar and personalize_handcrafted else {}
    with np.load(file_path, allow_pickle=False) as npz_file:
        feature_array = choose_feature_array(npz_file)
        label_array = choose_label_array(npz_file)

    if feature_array.ndim < 3:
        raise RuntimeError(f"Unsupported feature array in {file_path}: {feature_array.shape}")

    sequence_batch = coerce_sequence_batch(feature_array)
    sequence_batch = resize_sequence_batch(sequence_batch, sequence_length)

    if append_ear_mar and feature_array.ndim >= 4:
        handcrafted = build_handcrafted_features(normalize_landmark_batch(feature_array))
        if personalize_handcrafted:
            handcrafted = personalize_handcrafted_features(handcrafted, file_path, user_baselines)
        handcrafted = resize_sequence_batch(handcrafted, sequence_length)
        if handcrafted_only:
            sequence_batch = handcrafted.astype(np.float32)
        else:
            sequence_batch = np.concatenate([sequence_batch, handcrafted], axis=-1).astype(np.float32)

    return sequence_batch.astype(np.float32), label_array.astype(np.int64) if label_array is not None else None


def main() -> None:
    args = parse_args()
    split_dir = infer_split_dir(args.input, args.data_root)
    npz_paths = collect_npz_paths(args.input)
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found under {args.input}")

    label_map, mean, std, model = load_model_bundle(args.model_dir)
    inverse_map = {index: name for name, index in label_map.items()}
    threshold = args.threshold if args.threshold is not None else load_threshold(args.model_dir)
    if threshold is None:
        threshold = 0.5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_probs = []
    all_labels = []

    for file_path in npz_paths:
        sequences, labels = build_sequences_for_file(
            file_path=file_path,
            split_dir=split_dir,
            sequence_length=args.sequence_length,
            append_ear_mar=args.append_ear_mar,
            personalize_handcrafted=args.personalize_handcrafted,
            handcrafted_only=args.handcrafted_only,
        )
        sequences = normalize_sequences(sequences, mean, std)
        inputs = torch.from_numpy(sequences).to(device)
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        if labels is not None:
            all_labels.append(labels)

        preds = (probs[:, 1] >= threshold).astype(np.int64) if probs.shape[1] == 2 else np.argmax(probs, axis=1)
        counts = np.bincount(preds, minlength=len(label_map))
        print(f"{file_path.name}: {len(preds)} samples")
        for index, count in enumerate(counts):
            print(f"  predicted {inverse_map[index]}: {int(count)}")

    probabilities = np.concatenate(all_probs, axis=0)
    predictions = (probabilities[:, 1] >= threshold).astype(np.int64) if probabilities.shape[1] == 2 else np.argmax(probabilities, axis=1)

    print(f"\nUsed threshold: {threshold:.2f}")
    counts = np.bincount(predictions, minlength=len(label_map))
    for index, count in enumerate(counts):
        print(f"Total predicted {inverse_map[index]}: {int(count)}")

    if all_labels:
        labels = np.concatenate(all_labels, axis=0)
        metrics = build_metrics_from_predictions(labels, predictions)
        print("\nMetrics:")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
